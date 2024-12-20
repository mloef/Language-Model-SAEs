import os
from dataclasses import asdict
from typing import cast

import torch
import wandb
from datasets import Dataset, load_dataset, load_from_disk
import torch.distributed as dist
from torch.distributed.tensor import Replicate
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
)
from transformer_lens import HookedTransformer
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    ChameleonForConditionalGeneration,
    PreTrainedModel,
)
from concurrent.futures import ThreadPoolExecutor

from lm_saes.activation.token_source import MappedTokenSource

from .activation.activation_dataset import make_activation_dataset
from .activation.activation_source import CachedActivationSource
from .activation.activation_store import ActivationStore
from .analysis.features_to_logits import features_to_logits
from .analysis.sample_feature_activations import sample_feature_activations
from .config import (
    ActivationGenerationConfig,
    FeaturesDecoderConfig,
    LanguageModelCrossCoderTrainingConfig,
    LanguageModelSAEAnalysisConfig,
    LanguageModelSAEPruningConfig,
    LanguageModelSAERunnerConfig,
    LanguageModelSAETrainingConfig,
    SAEConfig,
)
from .crosscoder import CrossCoder
from .database import MongoClient
from .evals import run_evals
from .post_processing import post_process_topk_to_jumprelu_for_inference
from .sae import SparseAutoEncoder
from .sae_training import prune_sae, train_sae
from .utils.misc import is_master
from .utils.model import load_model

def get_sae_class(cfg: SAEConfig):
    class_dict = {
        'sparse_autoencoder': SparseAutoEncoder,
        'crosscoder': CrossCoder,
    }
    return class_dict[cfg.sae_type]


def language_model_sae_runner(cfg: LanguageModelSAETrainingConfig):
    if cfg.act_store.use_cached_activations:
        activation_source = CachedActivationSource(cfg.act_store)
        activation_store = ActivationStore(act_source=activation_source, cfg=cfg.act_store)
        model = None
    else:
        model = load_model(cfg.lm)
        activation_store = ActivationStore.from_config(model=model, cfg=cfg.act_store)

    if not cfg.finetuning and (
        cfg.sae.norm_activation == "dataset-wise"
        and cfg.sae.dataset_average_activation_norm is None
        or cfg.sae.init_decoder_norm is None
    ):
        sae = SparseAutoEncoder.from_initialization_searching(
            activation_store=activation_store,
            cfg=cfg,
        )
    else:
        sae = SparseAutoEncoder.from_config(cfg=cfg.sae)

        if cfg.finetuning:
            # Fine-tune SAE with frozen encoder weights and bias
            sae.train_finetune_for_suppression_parameters()

    if is_master():
        cfg.sae.save_hyperparameters(cfg.exp_result_path)
        cfg.lm.save_lm_config(cfg.exp_result_path)

    if cfg.wandb.log_to_wandb and is_master():
        wandb_config: dict = {
            **asdict(cfg),
            **asdict(cfg.sae),
            **asdict(cfg.lm),
        }
        del wandb_config["sae"]
        del wandb_config["lm"]
        wandb_run = wandb.init(
            project=cfg.wandb.wandb_project,
            config=wandb_config,
            name=cfg.wandb.exp_name,
            entity=cfg.wandb.wandb_entity,
            settings=wandb.Settings(x_disable_stats=True),
            mode=os.getenv("WANDB_MODE", "online"),
        )
        with open(os.path.join(cfg.exp_result_path, "train_wandb_id.txt"), "w") as f:
            f.write(wandb_run.id)
        wandb.watch(sae, log="all")

    # train SAE
    sae = train_sae(
        sae,
        activation_store,
        cfg,
        model,
    )

    if cfg.wandb.log_to_wandb and is_master():
        wandb.finish()

    return sae


def language_model_crosscoder_runner(cfg: LanguageModelCrossCoderTrainingConfig):
    activation_source = CachedActivationSource(cfg.act_store)
    activation_store = ActivationStore(act_source=activation_source, cfg=cfg.act_store)

    if not cfg.finetuning and (
        cfg.sae.norm_activation == "dataset-wise" and cfg.sae.dataset_average_activation_norm is None
    ):
        sae = CrossCoder.from_initialization_searching(
            activation_store=activation_store,
            cfg=cfg,
        )
    else:
        sae = CrossCoder.from_config(cfg=cfg.sae)

    sae.initialize_with_same_weight_across_layers()
    sae.train()
    # sae.search_for_enc_dec_norm_with_lowest_mse(
    #     activation_store=activation_store,
    #     cfg=cfg
    # )

    cfg.sae.save_hyperparameters(cfg.exp_result_path)

    if cfg.wandb.log_to_wandb and (cfg.wandb.log_on_every_rank or is_master()):
        wandb_config: dict = {
            **asdict(cfg),
            **asdict(cfg.sae),
            **asdict(cfg.lm),
        }
        del wandb_config["sae"]
        del wandb_config["lm"]
        wandb_run = wandb.init(
            project=cfg.wandb.wandb_project,
            config=wandb_config,
            name=cfg.wandb.exp_name,
            entity=cfg.wandb.wandb_entity,
            settings=wandb.Settings(x_disable_stats=True),
            mode=os.getenv("WANDB_MODE", "online"),
        )
        with open(os.path.join(cfg.exp_result_path, "train_wandb_id.txt"), "w") as f:
            f.write(wandb_run.id)
        wandb.watch(sae, log="all")

    # train SAE
    sae = train_sae(
        sae,
        activation_store,
        cfg,
    )

    if cfg.wandb.log_to_wandb and is_master():
        wandb.finish()

    return sae


def language_model_sae_prune_runner(cfg: LanguageModelSAEPruningConfig):
    cfg.sae.save_hyperparameters(os.path.join(cfg.exp_result_path))
    cfg.lm.save_lm_config(os.path.join(cfg.exp_result_path))
    sae = SparseAutoEncoder.from_config(cfg=cfg.sae)
    model = load_model(cfg.lm)
    activation_store = ActivationStore.from_config(model=model, cfg=cfg.act_store)
    if cfg.wandb.log_to_wandb and is_master():
        wandb_config: dict = {
            **asdict(cfg),
            **asdict(cfg.sae),
            **asdict(cfg.lm),
        }
        del wandb_config["sae"]
        del wandb_config["lm"]
        wandb_run = wandb.init(
            project=cfg.wandb.wandb_project,
            config=wandb_config,
            name=cfg.wandb.exp_name,
            entity=cfg.wandb.wandb_entity,
            settings=wandb.Settings(x_disable_stats=True),
            mode=os.getenv("WANDB_MODE", "online"),
        )
        with open(os.path.join(cfg.exp_result_path, "prune_wandb_id.txt"), "w") as f:
            f.write(wandb_run.id)

    sae = prune_sae(
        sae,
        activation_store,
        cfg,
    )

    result = run_evals(model, sae, activation_store, cfg, 0)

    # Print results in tabular format
    if is_master():
        for key, value in result.items():
            print(f"{key}: {value}")

    if cfg.wandb.log_to_wandb and is_master():
        wandb.finish()


def language_model_sae_eval_runner(cfg: LanguageModelSAERunnerConfig):
    sae = SparseAutoEncoder.from_config(cfg=cfg.sae)
    model = load_model(cfg.lm)
    activation_store = ActivationStore.from_config(model=model, cfg=cfg.act_store)

    if cfg.wandb.log_to_wandb and is_master():
        wandb_config: dict = {
            **asdict(cfg),
            **asdict(cfg.sae),
            **asdict(cfg.lm),
        }
        del wandb_config["sae"]
        del wandb_config["lm"]
        wandb_run = wandb.init(
            project=cfg.wandb.wandb_project,
            config=wandb_config,
            name=cfg.wandb.exp_name,
            entity=cfg.wandb.wandb_entity,
            settings=wandb.Settings(x_disable_stats=True),
            mode=os.getenv("WANDB_MODE", "online"),
        )
        with open(os.path.join(cfg.exp_result_path, "eval_wandb_id.txt"), "w") as f:
            f.write(wandb_run.id)

    result = run_evals(model, sae, activation_store, cfg, 0)

    # Print results in tabular format
    if is_master():
        for key, value in result.items():
            print(f"{key}: {value}")

    if cfg.wandb.log_to_wandb and is_master():
        wandb.finish()

    return sae


def activation_generation_runner(cfg: ActivationGenerationConfig):
    model = load_model(cfg.lm)

    make_activation_dataset(model, cfg)


def sample_feature_activations_runner(cfg: LanguageModelSAEAnalysisConfig):
    SaeClass = get_sae_class(cfg.sae)
    sae = SaeClass.from_config(cfg=cfg.sae)

    if cfg.sae.tp_size > 1:
        plan = {
            "encoder": ColwiseParallel(output_layouts=Replicate()),
            "decoder": RowwiseParallel(output_layouts=Replicate()),
        }
        if cfg.sae.use_glu_encoder:
            plan["encoder_glu"] = ColwiseParallel(output_layouts=Replicate())
        sae = cast(SaeClass, parallelize_module(sae, device_mesh=sae.device_mesh["tp"], parallelize_plan=plan))

    # JumpReLU need SAE decoder weight, so we currently do not remove it
    # sae.decoder.weight = None  # type: ignore[assignment]
    # torch.cuda.empty_cache()

    if cfg.act_store.use_cached_activations:
        activation_source = CachedActivationSource(cfg.act_store)
        activation_store = ActivationStore(act_source=activation_source, cfg=cfg.act_store)
        model = None
        dataset = None
        hf_tokenizer = AutoTokenizer.from_pretrained(
            (cfg.lm.model_name if cfg.lm.model_from_pretrained_path is None else cfg.lm.model_from_pretrained_path),
            trust_remote_code=True,
            use_fast=True,
            add_bos_token=True,
            local_files_only=True,
        )
        ignore_tokens = {
            hf_tokenizer.bos_token_id,
            hf_tokenizer.eos_token_id,
        }
    else:
        model = get_model(cfg.lm)
        assert len(cfg.dataset.dataset_path) == 1, "Only one dataset path is supported"
        if not cfg.dataset.is_dataset_on_disk:
            dataset = load_dataset(
                cfg.dataset.dataset_path[0], split="train", cache_dir=cfg.dataset.cache_dir, keep_in_memory=True
            )
        else:
            dataset = load_from_disk(cfg.dataset.dataset_path[0], keep_in_memory=True)
        dataset = cast(Dataset, dataset)
        dataset = dataset.with_format("torch", device=cfg.lm.device)
        activation_store = None
        ignore_tokens = {
            model.tokenizer.bos_token_id,
            model.tokenizer.eos_token_id,
            model.tokenizer.pad_token_id,
        }

    client = MongoClient(cfg.mongo.mongo_uri, cfg.mongo.mongo_db)
    if is_master():
        client.create_dictionary(cfg.exp_name, cfg.exp_result_path, cfg.sae.d_sae, cfg.exp_series)
    
    token_source = MappedTokenSource.from_config(model=model, cfg=cfg.dataset)

    for chunk_id in range(cfg.n_sae_chunks):
        result = sample_feature_activations(sae, model, token_source, cfg, chunk_id, cfg.n_sae_chunks)
        for i in range(len(result["index"].cpu().numpy().tolist())):
            client.update_feature(
                cfg.exp_name,
                result["index"][i].item(),
                {
                    "act_times": result["act_times"][i].item(),
                    "max_feature_acts": result["max_feature_acts"][i].item(),
                    "dataset": cfg.dataset.dataset_path,
                    "analysis": [
                        {
                            "name": v["name"],
                            "feature_acts": v["feature_acts"][i].cpu().float().numpy(),
                            "context_ids": v["context_ids"][i].cpu().numpy(),
                            "dataset_ids": v["dataset_ids"][i].cpu().numpy(),
                        }
                        for v in result["analysis"]
                    ],
                },
                dictionary_series=cfg.exp_series,
            )

        if is_master():
            parallel_write(cfg, client, result, os.cpu_count())
            
        del result
        torch.cuda.empty_cache()


@torch.no_grad()
def features_to_logits_runner(cfg: FeaturesDecoderConfig):
    sae = SparseAutoEncoder.from_config(cfg=cfg.sae)

    model = load_model(cfg.lm)

    result_dict = features_to_logits(sae, model, cfg)

    client = MongoClient(cfg.mongo.mongo_uri, cfg.mongo.mongo_db)

    for feature_index, logits in result_dict.items():
        sorted_indeces = torch.argsort(logits)
        top_negative_logits = logits[sorted_indeces[: cfg.top]].cpu().tolist()
        top_positive_logits = logits[sorted_indeces[-cfg.top :]].cpu().tolist()
        top_negative_ids = sorted_indeces[: cfg.top].tolist()
        top_positive_ids = sorted_indeces[-cfg.top :].tolist()
        top_negative_tokens = model.to_str_tokens(torch.tensor(top_negative_ids), prepend_bos=False)
        top_positive_tokens = model.to_str_tokens(torch.tensor(top_positive_ids), prepend_bos=False)
        counts, edges = torch.histogram(
            logits.cpu(), bins=60, range=(-60.0, 60.0)
        )  # Why logits.cpu():Could not run 'aten::histogram.bin_ct' with arguments from the 'CUDA' backend
        client.update_feature(
            cfg.exp_name,
            int(feature_index),
            {
                "logits": {
                    "top_negative": [
                        {"token_id": id, "logit": logit, "token": token}
                        for id, logit, token in zip(top_negative_ids, top_negative_logits, top_negative_tokens)
                    ],
                    "top_positive": [
                        {"token_id": id, "logit": logit, "token": token}
                        for id, logit, token in zip(top_positive_ids, top_positive_logits, top_positive_tokens)
                    ],
                    "histogram": {
                        "counts": counts.cpu().tolist(),
                        "edges": edges.cpu().tolist(),
                    },
                }
            },
            dictionary_series=cfg.exp_series,
        )


@torch.no_grad()
def post_process_topk_to_jumprelu_runner(cfg: LanguageModelSAERunnerConfig):
    sae = SparseAutoEncoder.from_config(cfg=cfg.sae)
    model = load_model(cfg.lm)

    activation_store = ActivationStore.from_config(model=model, cfg=cfg.act_store)
    post_process_topk_to_jumprelu_for_inference(sae, activation_store, cfg)
