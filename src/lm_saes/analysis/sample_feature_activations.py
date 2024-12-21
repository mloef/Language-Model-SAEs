# TODO: check if this is compatible for both cc and chameleon @xuyang


from typing import cast
import torch
import torch.distributed as dist
from einops import rearrange, repeat
from torch.distributed.tensor import DTensor
from tqdm import tqdm
from transformer_lens import HookedTransformer
from functools import partial

from lm_saes.activation.token_source import TokenSourceInfo, MappedTokenSource

from ..config import LanguageModelSAEAnalysisConfig
from ..sae import SparseAutoEncoder
from ..utils.misc import print_once
from ..utils.tensor_dict import concat_dict_of_tensor, sort_dict_of_tensor
from ..activation.activation_store import ActivationStore, ActivationSourceInfo, ActivationBatch


def _generator_on_the_fly(
        cfg: LanguageModelSAEAnalysisConfig,
        token_source: MappedTokenSource,
    ):
        while True:
            num_tokens_per_batch = cfg.dataset.store_batch_size * cfg.dataset.context_size
            batch = token_source.next(cfg.dataset.store_batch_size)
            assert batch is not None, "Token source is exhausted"
            tokens, sources = batch.tokens, batch.source_info

            _, cache = model.run_with_cache_until(
                tokens,
                names_filter=[cfg.sae.hook_point_in, cfg.sae.hook_point_out],
                until=cfg.sae.hook_point_out,
            )
            activation_in, _ = (
                cache[cfg.sae.hook_point_in],
                cache[cfg.sae.hook_point_out],
            )

            yield AnalysisBatch(
                tokens=tokens, 
                sources=sources, 
                unfiltered_activations=activation_in, 
                num_tokens_per_batch=num_tokens_per_batch,
            )

def _generator_with_cached_store(
        cfg: LanguageModelSAEAnalysisConfig,
        activation_store: ActivationStore,
    ):
        context_idx = 0
        while True:
            num_tokens_per_batch = cfg.dataset.store_batch_size * cfg.dataset.context_size
            batch = activation_store.next(batch_size=num_tokens_per_batch)
            assert batch is not None, "Activation store is empty"

            print(batch)

            yield AnalysisBatch(
                tokens=batch['context'].view(
                    cfg.dataset.store_batch_size,
                    cfg.dataset.context_size,
                ),
                sources=[
                    TokenSourceInfo(
                        dataset_idx=0,  # Currently we do not consider multiple datasets
                        context_idx=context_idx + i
                    )
                    for i in range(len(batch))
                ],
                unfiltered_activations=batch[cfg.act_store.hook_points[0]].view(
                    cfg.dataset.store_batch_size, 
                    cfg.dataset.context_size, 
                    cfg.lm.d_model,
                ), 
                num_tokens_per_batch=num_tokens_per_batch
            )

            context_idx += len(batch)


@torch.no_grad()
def sample_feature_activations(
    sae: SparseAutoEncoder,
    model: HookedTransformer,
    token_source: MappedTokenSource,
    activation_store: ActivationStore,
    cfg: LanguageModelSAEAnalysisConfig,
    sae_chunk_id: int = 0,
    n_sae_chunks: int = 1,  # By default, we do not chunk the SAE. When the model & SAE is large, we can chunk the SAE to save memory.
    ignore_tokens: set[int | None] = set(),
):
    generate_activations_on_the_fly = token_source is not None
    assert generate_activations_on_the_fly ^ (activation_store is not None), \
        "Must provide either model & dataset or cached activation store"

    if sae.cfg.ddp_size > 1:
        raise ValueError("Sampling feature activations does not support DDP yet")
    assert cfg.sae.d_sae is not None  # Make mypy happy

    total_analyzing_tokens = cfg.total_analyzing_tokens
    total_analyzing_steps = total_analyzing_tokens // cfg.dataset.store_batch_size // cfg.dataset.context_size

    print_once(f"Total Analyzing Tokens: {total_analyzing_tokens}")
    print_once(f"Total Analyzing Steps: {total_analyzing_steps}")

    n_training_steps = 0
    n_training_tokens = 0

    sae.eval()

    pbar = tqdm(
        total=total_analyzing_tokens,
        desc=f"Sampling activations of chunk {sae_chunk_id} of {n_sae_chunks}",
        smoothing=0.01,
    )

    d_sae = cfg.sae.d_sae // n_sae_chunks
    assert d_sae // cfg.sae.tp_size * cfg.sae.tp_size == d_sae, "d_sae must be divisible by tp_size"
    d_sae //= cfg.sae.tp_size

    rank = dist.get_rank() if cfg.sae.tp_size > 1 else 0
    start_index = sae_chunk_id * d_sae * cfg.sae.tp_size + d_sae * rank
    end_index = sae_chunk_id * d_sae * cfg.sae.tp_size + d_sae * (rank + 1)

    sample_result = {
        k: {
            "elt": torch.empty((0, d_sae), dtype=cfg.sae.dtype, device=cfg.sae.device),
            "feature_acts": torch.empty(
                (0, d_sae, cfg.dataset.context_size),
                dtype=cfg.sae.dtype,
                device=cfg.sae.device,
            ),
            "context_ids": torch.empty(
                (0, d_sae),
                dtype=torch.int32,
                device=cfg.sae.device,
            ),
            "dataset_ids": torch.empty(
                (0, d_sae),
                dtype=torch.int32,
                device=cfg.sae.device,
            ),
        }
        for k in cfg.subsample.keys()
    }
    act_times = torch.zeros((d_sae,), dtype=torch.long, device=cfg.sae.device)
    max_feature_acts = torch.zeros((d_sae,), dtype=cfg.sae.dtype, device=cfg.sae.device)


    if generate_activations_on_the_fly:
        token_activation_generator = _generator_on_the_fly(
            cfg=cfg,
            token_source=token_source,
        )
    else:   
        token_activation_generator = _generator_with_cached_store(
            cfg=cfg,
            activation_store=activation_store,
        )

    for batch_idx, batch in enumerate(token_activation_generator):
        if len(ignore_tokens) > 0:
            filter_mask = torch.any(
                torch.stack(
                    [batch.tokens == ignore_token for ignore_token in ignore_tokens], dim=0
                ),
                dim=0,
            )

        feature_acts = sae.encode(batch.activation_in)[..., start_index:end_index]
        if isinstance(feature_acts, DTensor):
            feature_acts = feature_acts.to_local()

        feature_acts[filter_mask] = 0
        act_times += feature_acts.gt(0.0).sum(dim=[0, 1])

        for name in cfg.subsample.keys():
            if cfg.enable_sampling:
                weights = feature_acts.clamp(min=0.0).pow(cfg.sample_weight_exponent).max(dim=1).values
                elt = torch.rand(batch.tokens.size(0), d_sae, device=cfg.sae.device, dtype=cfg.sae.dtype).log() / weights
                elt[weights == 0.0] = -torch.inf
            else:
                elt = feature_acts.clamp(min=0.0).max(dim=1).values

            elt[
                feature_acts.max(dim=1).values > max_feature_acts.unsqueeze(0) * cfg.subsample[name]["proportion"]
            ] = -torch.inf

            if (
                sample_result[name]["elt"].size(0) > 0
                and (elt.max(dim=0).values <= sample_result[name]["elt"][-1]).all()
            ):
                continue

            sample_result[name] = concat_dict_of_tensor(
                sample_result[name],
                {
                    "elt": elt,
                    "feature_acts": rearrange(
                        feature_acts,
                        "batch_size context_size d_sae -> batch_size d_sae context_size",
                    ),
                    "context_ids": repeat(
                        torch.tensor([source.context_idx for source in batch.sources], device=cfg.sae.device),
                        "batch_size -> batch_size d_sae",
                        d_sae=d_sae,
                    ),
                    "dataset_ids": repeat(
                        torch.tensor([source.dataset_idx for source in batch.sources], device=cfg.sae.device),
                        "batch_size -> batch_size d_sae",
                        d_sae=d_sae,
                    ),
                },
                dim=0,
            )

            sample_result[name] = sort_dict_of_tensor(sample_result[name], sort_dim=0, sort_key="elt", descending=True)
            sample_result[name] = {k: v[: cfg.subsample[name]["n_samples"]] for k, v in sample_result[name].items()}

        max_feature_acts = torch.max(max_feature_acts, feature_acts.max(dim=0).values.max(dim=0).values)

        n_training_tokens += batch.num_tokens_per_batch
        n_training_steps += 1

        pbar.update(batch.num_tokens_per_batch)

    pbar.close()

    sample_result = {
        k1: {k2: rearrange(v2, "n_samples d_sae ... -> d_sae n_samples ...") for k2, v2 in v1.items()}
        for k1, v1 in sample_result.items()
    }

    result = {
        "index": torch.arange(start_index, end_index, device=cfg.sae.device, dtype=torch.int32),
        "act_times": act_times,
        "max_feature_acts": max_feature_acts,
        "analysis": [
            {
                "name": k,
                "feature_acts": v["feature_acts"],
                "context_ids": v["context_ids"],
                "dataset_ids": v["dataset_ids"],
            }
            for k, v in sample_result.items()
        ],
    }

    return result
