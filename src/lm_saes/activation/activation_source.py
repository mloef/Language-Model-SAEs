import random
from abc import ABC
from typing import Dict, NamedTuple
from dataclasses import dataclass

import torch
import torch.distributed as dist
from einops import rearrange
from transformer_lens import HookedTransformer
from threading import Thread
import queue

from ..config import ActivationStoreConfig
from .token_source import TokenSource
from .utils import list_activation_chunks, load_activation_chunk


class ActivationSourceInfo(NamedTuple):
    dataset_idx: int
    context_idx: int


@dataclass
class ActivationBatch:
    context: torch.Tensor
    sources: ActivationSourceInfo
    activations: torch.Tensor

    def __len__(self):
        return len(self.context)
    
    def fetch_a_batch(self, batch_size: int):
        result = ActivationBatch(
            context=self.context[:batch_size],
            sources=self.sources[:batch_size],
            activations=self.activations[:batch_size],
        )
        residual = ActivationBatch(
            context=self.context[batch_size:],
            sources=self.sources[batch_size:],
            activations=self.activations[batch_size:],
        )
        return result, residual
    
    def merge_buffer(self, buffer: ActivationBatch):
        self.context = torch.cat([self.context, buffer.context], dim=0)
        self.sources = self.sources + buffer.sources
        self.activations = torch.cat([self.activations, buffer.activations], dim=0)

        return self


class ActivationSource(ABC):
    def next(self) -> Dict[str, torch.Tensor] | None:
        """
        Get the next batch of activations.

        Returns:
            A dictionary where the keys are the names of the activations and the values are tensors of shape (batch_size, d_model). If there are no more activations, return None.
        """
        raise NotImplementedError

    def next_tokens(self, batch_size: int) -> torch.Tensor | None:
        """
        Get the next batch of tokens.

        Returns:
            A tensor of shape (batch_size, seq_len) where seq_len is the length of the sequence.
        """
        raise NotImplementedError


class TokenActivationSource(ActivationSource):
    """
    An activation source that generates activations from a token source.
    """

    def __init__(self, model: HookedTransformer, cfg: ActivationStoreConfig):
        self.token_source = TokenSource.from_config(model=model, cfg=cfg.dataset)
        self.model = model
        assert model.tokenizer is not None, "Tokenizer is not set"
        self.tokenizer = model.tokenizer
        self.cfg = cfg

    def next(self) -> Dict[str, torch.Tensor] | None:
        tokens = self.next_tokens(self.cfg.dataset.store_batch_size)

        if tokens is None:
            return None
        with torch.no_grad():
            _, cache = self.model.run_with_cache_until(
                tokens, names_filter=self.cfg.hook_points, until=self.cfg.hook_points[-1]
            )

            filter_mask = torch.logical_and(
                tokens != self.tokenizer.eos_token_id, tokens != self.tokenizer.pad_token_id
            )
            filter_mask = torch.logical_and(filter_mask, tokens != self.tokenizer.bos_token_id)

            filter_mask = rearrange(filter_mask, "b l -> (b l)")

            ret = {
                k: rearrange(cache[k].to(dtype=self.cfg.dtype, device=self.cfg.device), "b l d -> (b l) d")[filter_mask]
                for k in self.cfg.hook_points
            }

            return ret

    def next_tokens(self, batch_size: int) -> torch.Tensor | None:
        return self.token_source.next(batch_size)


class CachedActivationSource(ActivationSource):
    def __init__(self, cfg: ActivationStoreConfig):
        self.cfg = cfg
        self.sample_probs = cfg.cache_sample_probs

        assert cfg.use_cached_activations
        if len(cfg.hook_points) > 1:
            print(
                "CachedActivationSource only supports one hook point, but %d hook points are provided. Only the first hook point will be used.",
                len(cfg.hook_points),
            )
        self.hook_point = cfg.hook_points[0]
        self.chunk_paths: list[list[str]] = [
            list_activation_chunks(cached_activations_path, self.hook_point)
            for cached_activations_path in cfg.cached_activations_path
        ]
        self.chunk_buffer = {}  # n_tokens_in_buffer
        if cfg.ddp_size > 1:
            self.chunk_paths = [
                [p for i, p in enumerate(chunk_paths) if i % dist.get_world_size() == dist.get_rank()]
                for chunk_paths in self.chunk_paths
            ]
        if cfg.shuffle_activations:
            for num in range(len(self.chunk_paths)):
                random.shuffle(self.chunk_paths[num])

        self.token_buffer = torch.empty((0, cfg.dataset.context_size), dtype=torch.long, device=cfg.device)

    def load_chunk_into_buffer(self, dataset_id, chunk_path: list[str], ban_token_list=None):
        if dataset_id not in self.chunk_buffer:
            self.chunk_buffer[dataset_id] = torch.empty(
                (0, self.cfg.lm.d_model), dtype=self.cfg.dtype, device=self.cfg.device
            )
        to_fill_length = self.cfg.n_tokens_in_buffer // len(self.chunk_paths) - self.chunk_buffer[dataset_id].size(0)  # Why // len(self.chunk_paths)
        while to_fill_length > 0 and len(chunk_path) > 0:
            chunk = load_activation_chunk(chunk_path.pop(), self.cfg.device)
            with_context = len(chunk["activation"].size()) == 3
            activation = chunk["activation"]
            if with_context:
                chunk["context"] = chunk["context"].to(dtype=torch.long, device=self.cfg.device, non_blocking=True)
                not_ban_token = torch.isin(
                    rearrange(chunk["context"], "b l -> (b l)"),
                    torch.tensor(ban_token_list, device=self.cfg.device),
                    invert=True,
                )
                activation = rearrange(chunk["activation"], "b l d -> (b l) d")[not_ban_token]
            self.chunk_buffer[dataset_id] = torch.cat([self.chunk_buffer[dataset_id], activation], dim=0)
            to_fill_length -= activation.size(0)
        return chunk_path

    def next(self) -> Dict[str, torch.Tensor] | None:
        for i, chunk_paths in enumerate(self.chunk_paths):
            self.sample_probs[i] = 0 if len(chunk_paths) == 0 else self.sample_probs[i]
        self.sample_probs = [p / sum(self.sample_probs) for p in self.sample_probs]

        for i, chunk_paths in enumerate(self.chunk_paths):
            self.chunk_paths[i] = self.load_chunk_into_buffer(i, chunk_paths, self.cfg.ban_token_list[i])

        next_length_list = [
            min(int(self.sample_probs[i] * self.cfg.n_tokens_in_buffer), self.chunk_buffer[i].size(0))
            for i in range(len(self.chunk_paths))
        ]

        ret = {
            self.hook_point: torch.cat(
                [
                    self.chunk_buffer[i][: next_length_list[i]] 
                    for i in range(len(self.chunk_paths))
                ],
                dim=0
            )
        }
        return ret

    def next_tokens(self, batch_size: int) -> torch.Tensor | None:
        if self.token_buffer.size(0) < batch_size:
            chunk = self._load_next_chunk()
            if chunk is None:
                return None
            self.token_buffer = torch.cat([self.token_buffer, chunk["context"]], dim=0)
        ret = self.token_buffer[:batch_size]
        self.token_buffer = self.token_buffer[batch_size:]
        return ret


class MappedActivationSource(ActivationSource):
    def __init__(self, cfg: ActivationStoreConfig):
        self.cfg = cfg
        self.sample_probs = cfg.cache_sample_probs

        assert cfg.use_cached_activations

        if len(cfg.hook_points) > 1:
            print(
                "CachedActivationSource only supports one hook point, but %d hook points are provided. Only the first hook point will be used.",
                len(cfg.hook_points),
            )

        self.hook_point = cfg.hook_points[0]
        self.chunk_paths: list[list[str]] = [
            list_activation_chunks(cached_activations_path, self.hook_point)
            for cached_activations_path in cfg.cached_activations_path
        ]
        self.chunk_buffer = {}  # n_tokens_in_buffer
        if cfg.ddp_size > 1:
            raise NotImplementedError('Not implemented')
        if cfg.shuffle_activations:
            raise ValueError('`shuffle_activations` is not recommended for MappedActivationSource in analysis')

        self.token_buffer = torch.empty((0, cfg.dataset.context_size), dtype=torch.long, device=cfg.device)
    
    def load_chunk_into_buffer(self, dataset_id, chunk_path: list[str], ban_token_list=None):
        if dataset_id not in self.chunk_buffer:
            self.chunk_buffer[dataset_id] = ActivationBatch(
                tokens=torch.empty(
                    (0, self.cfg.dataset.context_size), 
                    dtype=self.cfg.dtype, 
                    device=self.cfg.device,
                ),
                sources=list(),
                activations=torch.empty(
                    (0, self.cfg.dataset.context_size, self.cfg.lm.d_model), 
                    dtype=self.cfg.dtype, 
                    device=self.cfg.device,
                ),
            )
        to_fill_length = self.cfg.n_tokens_in_buffer // len(self.chunk_paths) - self.chunk_buffer[dataset_id].size(0)
        while to_fill_length > 0 and len(chunk_path) > 0:
            chunk = load_activation_chunk(chunk_path.pop(), self.cfg.device)
            with_context = len(chunk["activation"].size()) == 3
            if not with_context:
                raise ValueError(
                    f'The shape of saved activation is advised to be `(chunk_size, seq_len, d_model)`, got `{chunk["activation"].size()}`'
                )
            activation = chunk["activation"].to(dtype=torch.long, device=self.cfg.device, non_blocking=True)
            context = chunk["context"].to(dtype=torch.long, device=self.cfg.device, non_blocking=True)
            source_info = [
                ActivationSourceInfo(
                    dataset_idx=dataset_id
                    context_idx=chunk["context_idx"][i].item()
                )
                for i in range(activation.size(0))
            ]

            self.chunk_buffer[dataset_id].merge_buffer(
                ActivationBatch(
                    context=context,
                    sources=source_info,
                    activations=activation,
                ),
            )
            to_fill_length -= activation.size(0)
        return chunk_path
    
    def next(self) -> Dict[str, torch.Tensor] | None:
        for i, chunk_paths in enumerate(self.chunk_paths):
            self.sample_probs[i] = 0 if len(chunk_paths) == 0 else self.sample_probs[i]
        self.sample_probs = [p / sum(self.sample_probs) for p in self.sample_probs]

        for i, chunk_paths in enumerate(self.chunk_paths):
            self.chunk_paths[i] = self.load_chunk_into_buffer(i, chunk_paths, self.cfg.ban_token_list[i])

        next_length_list = [
            min(int(self.sample_probs[i] * self.cfg.n_tokens_in_buffer), self.chunk_buffer[i].size(0))
            for i in range(len(self.chunk_paths))
        ]

        ret = {
            self.hook_point: torch.cat(
                [
                    self.chunk_buffer[i][: next_length_list[i]] 
                    for i in range(len(self.chunk_paths))
                ],
                dim=0
            ),
            'context': 
        }
        return ret