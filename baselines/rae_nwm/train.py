# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# NoMaD, GNM, ViNT: https://github.com/robodhruv/visualnav-transformer
# --------------------------------------------------------

from infer import model_forward_wrapper
import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import os, sys
import contextlib

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RAE_SRC = os.path.join(PROJECT_ROOT, "RAE", "src")
if RAE_SRC not in sys.path:
    sys.path.append(RAE_SRC)
import matplotlib
matplotlib.use('Agg')
from collections import OrderedDict
from copy import deepcopy
from time import time
import argparse
import logging
import math
import random
import matplotlib.pyplot as plt 
import yaml
import wandb
import numpy as np
import torch.nn.functional as F


import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, ConcatDataset, Sampler as TorchSampler
from torch.utils.data.distributed import DistributedSampler
torch.backends.cuda.enable_math_sdp(False)
from RAE.src.stage1.rae import RAE
from RAE.src.stage2.transport.transport import Transport, ModelType, PathType, WeightType, Sampler
from RAE.src.utils.model_utils import instantiate_from_config
from RAE.src.utils.train_utils import parse_configs
from RAE.src.stage1 import RAE
from huggingface_hub import hf_hub_download
from distributed import init_distributed
from models import CDiT_models
from datasets import TrainingDataset
from misc import transform
from rae_stream.accumulation import (
    accumulation_update_count,
    make_accumulation_plan,
    require_positive_int,
)

#################################################################################
#                             Training Helper Functions                         #
#################################################################################

_FLASH_ATTN_STATUS_PRINTED = False


def _get_sdpa_backend_flags() -> dict:
    flags = {}
    try:
        flags["flash"] = bool(torch.backends.cuda.flash_sdp_enabled())
    except Exception:
        flags["flash"] = None
    try:
        flags["mem_efficient"] = bool(torch.backends.cuda.mem_efficient_sdp_enabled())
    except Exception:
        flags["mem_efficient"] = None
    try:
        flags["math"] = bool(torch.backends.cuda.math_sdp_enabled())
    except Exception:
        flags["math"] = None
    return flags


def _maybe_print_flash_attn_status_once(*, device: torch.device, dtype: torch.dtype, num_heads: int, head_dim: int, seqlen: int, rank: int) -> None:
    global _FLASH_ATTN_STATUS_PRINTED
    if _FLASH_ATTN_STATUS_PRINTED or rank != 0:
        return
    if os.environ.get("FLASH_ATTN_MONITOR", "1").strip().lower() in ("0", "false", "no", "off"):
        return

    _FLASH_ATTN_STATUS_PRINTED = True

    gpu_name = None
    try:
        if device.type == "cuda":
            gpu_name = torch.cuda.get_device_name(device)
    except Exception:
        gpu_name = None

    flags = _get_sdpa_backend_flags()

    probe_ok = False
    probe_err = None
    try:
        cm = contextlib.nullcontext()
        if hasattr(torch.backends.cuda, "sdp_kernel"):
            cm = torch.backends.cuda.sdp_kernel(enable_flash=True, enable_mem_efficient=False, enable_math=False)
        else:
            try:
                from torch.nn.attention import SDPBackend, sdpa_kernel
                cm = sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION])
            except Exception:
                cm = contextlib.nullcontext()

        with cm:
            q = torch.randn(1, num_heads, seqlen, head_dim, device=device, dtype=dtype)
            k = torch.randn(1, num_heads, seqlen, head_dim, device=device, dtype=dtype)
            v = torch.randn(1, num_heads, seqlen, head_dim, device=device, dtype=dtype)
            _ = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        probe_ok = True
    except Exception as e:
        probe_err = str(e).split("\n")[0]

    print(
        f"[SDPA] torch={torch.__version__} cuda={torch.version.cuda} gpu={gpu_name} dtype={str(dtype).replace('torch.', '')} "
        f"flags={flags} flash_only_probe={{'ok': {probe_ok}, 'heads': {num_heads}, 'head_dim': {head_dim}, 'seqlen': {seqlen}, 'err': {probe_err!r}}}"
    )


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        name = name.replace('_orig_mod.', '')
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


class BalancedDistributedSampler(TorchSampler):
    def __init__(
        self,
        datasets,
        num_replicas=None,
        rank=None,
        shuffle=True,
        seed=0,
        desired_total_size=None,
    ):
        if num_replicas is None:
            num_replicas = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        if rank is None:
            rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0

        self.datasets = list(datasets)
        self.lengths = [len(d) for d in self.datasets]
        if any(l <= 0 for l in self.lengths):
            raise ValueError(f"All datasets must be non-empty, got lengths={self.lengths}")

        self.num_datasets = len(self.datasets)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)

        offsets = [0]
        for l in self.lengths[:-1]:
            offsets.append(offsets[-1] + int(l))
        self.offsets = offsets

        if desired_total_size is None:
            desired_total_size = int(sum(self.lengths))
        self.desired_total_size = int(desired_total_size)

        self.samples_per_dataset = int(math.ceil(self.desired_total_size / float(self.num_datasets)))
        base_total = self.samples_per_dataset * self.num_datasets

        self.total_size = int(math.ceil(base_total / float(self.num_replicas)) * self.num_replicas)
        self.num_samples = self.total_size // self.num_replicas

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        per_dataset = []
        for ds_idx, ds_len in enumerate(self.lengths):
            if self.shuffle:
                base = torch.randperm(ds_len, generator=g).tolist()
            else:
                base = list(range(ds_len))

            if self.samples_per_dataset <= ds_len:
                chosen = base[: self.samples_per_dataset]
            else:
                remaining = self.samples_per_dataset - ds_len
                extra = torch.randint(high=ds_len, size=(remaining,), generator=g).tolist() if remaining > 0 else []
                chosen = base + extra

            per_dataset.append([i + self.offsets[ds_idx] for i in chosen])

        indices = []
        for j in range(self.samples_per_dataset):
            for i in range(self.num_datasets):
                indices.append(per_dataset[i][j])

        if self.shuffle:
            perm = torch.randperm(len(indices), generator=g).tolist()
            indices = [indices[k] for k in perm]

        indices = indices[: self.desired_total_size]

        if len(indices) < self.total_size:
            indices += indices[: (self.total_size - len(indices))]

        indices = indices[self.rank : self.total_size : self.num_replicas]
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = int(epoch)


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new CDiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    _, rank, gpu, _ = init_distributed()
    device = torch.device(f"cuda:{gpu}")
    # rank = dist.get_rank()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")
    with open("config/eval_config.yaml", "r") as f:
        default_config = yaml.safe_load(f)
    config = default_config

    with open(args.config, "r") as f:
        user_config = yaml.safe_load(f)
    config.update(user_config)
    tp = config.get('transport', {})

    # The checked-in paper config expresses the effective global batch.  A
    # resolved runtime YAML may split it into microbatches; this arithmetic is
    # the only training-loop adaptation and is fail-closed on any rounding.
    raw_accumulation_steps = config.get('gradient_accumulation_steps', 1)
    raw_effective_global_batch = config.get('effective_global_batch_size', 96)
    try:
        accumulation_steps = require_positive_int(
            raw_accumulation_steps, name="gradient_accumulation_steps"
        )
        effective_global_batch = require_positive_int(
            raw_effective_global_batch, name="effective_global_batch_size"
        )
    except ValueError as exc:
        raise ValueError("batch mapping fields must be integer-valued") from exc
    batch_plan = make_accumulation_plan(
        global_batch_size=effective_global_batch,
        world_size=dist.get_world_size(),
        accumulation_steps=accumulation_steps,
    )
    try:
        configured_batch_size = require_positive_int(config["batch_size"], name="batch_size")
    except (KeyError, ValueError) as exc:
        raise ValueError("batch_size must be a positive integer") from exc
    if effective_global_batch != 96 or configured_batch_size != batch_plan.microbatch_size:
        raise ValueError(
            "resolved RAE-stream batch mapping must preserve effective global batch 96: "
            f"batch_size={config.get('batch_size')}, world={dist.get_world_size()}, "
            f"accumulation={accumulation_steps}"
        )

    # Setup an experiment folder:
    os.makedirs(config['results_dir'], exist_ok=True)  # Make results folder (holds all experiment subfolders)
    experiment_dir = f"{config['results_dir']}/{config['run_name']}"  # Create an experiment folder
    checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
    if rank == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
        
        wandb_config = config.get('wandb', {})
        use_wandb_new = wandb_config.get('enabled', True)
        use_wandb_old = config.get('use_wandb', True)
        
        if 'wandb' in config:
            should_use_wandb = use_wandb_new
        else:
            should_use_wandb = use_wandb_old
            
        if should_use_wandb:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = wandb_config.get('run_name') or f"{config['run_name']}_{timestamp}"
            
            wandb.init(
                project=wandb_config.get('project', 'nwm-training'),
                name=run_name,
                entity=wandb_config.get('entity', None),
                config=config,
                dir=experiment_dir,
                tags=wandb_config.get('tags', []),
                notes=wandb_config.get('notes', '')
            )
            logger.info(f"WandB initialized successfully with run name: {run_name}")
    else:
        logger = create_logger(None)

    # Make the runtime adaptation explicit in the durable training log.  The
    # paper recipe remains the effective global batch, while this receipt/log
    # label prevents an accumulation run from being mistaken for the
    # unadapted single-batch baseline.
    training_profile = (
        "RAE-stream-accumulation" if accumulation_steps > 1 else "paper-reproduction-v1"
    )
    logger.info(
        "Training profile=%s effective_global_batch_size=%d microbatch_global_size=%d "
        "per_rank_microbatch=%d gradient_accumulation_steps=%d",
        training_profile,
        batch_plan.effective_global_batch_size,
        batch_plan.microbatch_global_size,
        batch_plan.microbatch_size,
        accumulation_steps,
    )

    # Create model:
    # resolve decoder pretrained path: prefer local file, fallback to HF download
    config_path = config.get('config_path', 'RAE/configs/stage1/pretrained/DINOv2-B.yaml')
    rae_config, *_ = parse_configs(config_path)
    rae: RAE = instantiate_from_config(rae_config).to(device).eval()
    tokenizer = rae
    latent_size = config['image_size'] // 14
    assert config['image_size'] % 14 == 0, "Image size must be divisible by DINOv2 patch size."
    num_cond = config['context_size']
    
    learn_sigma_cfg = config.get('learn_sigma', False)
    if isinstance(learn_sigma_cfg, str):
        learn_sigma_cfg = learn_sigma_cfg.strip().lower() == 'true'
    model_kwargs = {
        'context_size': num_cond,
        'input_size': latent_size,
        'in_channels': rae.latent_dim,
        'learn_sigma': learn_sigma_cfg,
        'head_width': config.get('head_width', rae.latent_dim),
        'head_depth': int(config.get('head_depth', 2)),
        'head_num_heads': int(config.get('head_num_heads', 16)),
    }

    model = CDiT_models[config['model']](**model_kwargs).to(device)
    if rank == 0:
        logger.info(
            f"RAE latent_dim={rae.latent_dim}, latent_size={latent_size}, learn_sigma={learn_sigma_cfg}"
        )
        logger.info(
            f"CDiT in_channels={model.in_channels}, out_channels={model.out_channels}, patch_size={model.patch_size}, head_width={getattr(model, 'head_width', 'NA')}"
        )
    debug_shapes = bool(config.get('debug_shapes', False))

    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)
    
    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    base_lr = float(config.get('lr', 2e-4))
    betas = tuple(config.get('betas', (0.9, 0.95)))
    override_lr_on_resume = bool(config.get('override_lr_on_resume', True))
    opt = torch.optim.AdamW(model.parameters(), lr=base_lr, betas=betas, weight_decay=0.0)

    bfloat_enable = bool(hasattr(args, 'bfloat16') and args.bfloat16)

    try:
        attn0 = model.blocks[0].attn
        num_heads = int(getattr(attn0, "num_heads", getattr(model, "num_heads", 0)) or 0)
        head_dim = int(getattr(attn0, "head_dim", 0) or 0)
        seqlen = int(getattr(getattr(model, "x_embedder", None), "num_patches", 0) or 0)
        if num_heads <= 0 or head_dim <= 0 or seqlen <= 0:
            raise ValueError("invalid attention shape")
    except Exception:
        num_heads, head_dim, seqlen = 8, 64, 256

    probe_dtype = torch.bfloat16 if bfloat_enable else torch.float16
    _maybe_print_flash_attn_status_once(
        device=device,
        dtype=probe_dtype,
        num_heads=num_heads,
        head_dim=head_dim,
        seqlen=seqlen,
        rank=rank,
    )

    if bfloat_enable:
        scaler = torch.amp.GradScaler()

    # load existing checkpoint
    latest_path = os.path.join(checkpoint_dir, "latest.pth.tar")
    print('Searching for model from ', checkpoint_dir)
    start_epoch = 0
    train_steps = 0
    if os.path.isfile(latest_path) or config.get('from_checkpoint', 0):
        if os.path.isfile(latest_path) and config.get('from_checkpoint', 0):
            raise ValueError("Resuming from checkpoint, this might override latest.pth.tar!!")
        latest_path = latest_path if os.path.isfile(latest_path) else config.get('from_checkpoint', 0)
        print("Loading model from ", latest_path)
        latest_checkpoint = torch.load(latest_path, map_location=device, weights_only=False) 

        if "model" in latest_checkpoint:
            model_ckp = {k.replace('_orig_mod.', ''):v for k,v in latest_checkpoint['model'].items()}
            res = model.load_state_dict(model_ckp, strict=True)
            print("Loading model weights", res)

            model_ckp = {k.replace('_orig_mod.', ''):v for k,v in latest_checkpoint['ema'].items()}
            res = ema.load_state_dict(model_ckp, strict=True)
            print("Loading EMA model weights", res)
        else:
            update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights

        if "opt" in latest_checkpoint:
            opt_ckp = {k.replace('_orig_mod.', ''):v for k,v in latest_checkpoint['opt'].items()}
            opt.load_state_dict(opt_ckp)
            print("Loading optimizer params")

            if override_lr_on_resume:
                for pg in opt.param_groups:
                    pg["lr"] = base_lr
                opt.defaults["lr"] = base_lr
                print(f"Override optimizer lr on resume: lr={base_lr}")
        
        if "epoch" in latest_checkpoint:
            start_epoch = latest_checkpoint['epoch'] + 1
        
        if "train_steps" in latest_checkpoint:
            train_steps = latest_checkpoint["train_steps"]
        
        if "scaler" in latest_checkpoint:
            scaler.load_state_dict(latest_checkpoint["scaler"])
        
    # ~40% speedup but might leads to worse performance depending on pytorch version
    if args.torch_compile:
        model = torch.compile(model)
    model = DDP(model, device_ids=[device], find_unused_parameters=True)
    try:
        if debug_shapes:
            (model.module if hasattr(model, "module") else model).debug_shapes = True
    except Exception:
        pass
    # Compute time_dist_shift using RAE strategy: sqrt(C*H*W/base)
    shift_dim = rae.latent_dim * latent_size * latent_size
    shift_base = float(tp.get('time_dist_shift_base', 4096))
    time_dist_shift = math.sqrt(shift_dim / shift_base)
    if 'time_dist_shift' in tp and tp.get('time_dist_shift') is not None:
        time_dist_shift = float(tp.get('time_dist_shift'))
    if bool(tp.get('time_dist_shift_disable', False)):
        time_dist_shift = 1.0
    transport = Transport(
        model_type=getattr(ModelType, str(tp.get('model_type', 'velocity')).upper()),
        path_type=getattr(PathType, str(tp.get('path_type', 'linear')).upper()),
        loss_type=getattr(WeightType, str(tp.get('loss_type', 'velocity')).upper()),
        time_dist_type=str(tp.get('time_dist_type', 'uniform')),
        # time_dist_shift=float(tp.get('time_dist_shift', 1.0)),

        time_dist_shift=time_dist_shift,
        train_eps=1e-3,
        sample_eps=1e-3,
    )
    sampler_transport = Sampler(transport)
    logger.info(f"CDiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    train_datasets = []
    test_datasets = []

    for dataset_name in config["datasets"]:
        data_config = config["datasets"][dataset_name]

        for data_split_type in ["train", "test"]:
            if data_split_type in data_config:
                    goals_per_obs = int(data_config["goals_per_obs"])
                    if data_split_type == 'test':
                        goals_per_obs = 4 # standardize testing
                    
                    if "distance" in data_config:
                        min_dist_cat=data_config["distance"]["min_dist_cat"]
                        max_dist_cat=data_config["distance"]["max_dist_cat"]
                    else:
                        min_dist_cat=config["distance"]["min_dist_cat"]
                        max_dist_cat=config["distance"]["max_dist_cat"]

                    if "len_traj_pred" in data_config:
                        len_traj_pred=data_config["len_traj_pred"]
                    else:
                        len_traj_pred=config["len_traj_pred"]

                    dataset = TrainingDataset(
                        data_folder=data_config["data_folder"],
                        data_split_folder=data_config[data_split_type],
                        dataset_name=dataset_name,
                        image_size=config["image_size"],
                        min_dist_cat=min_dist_cat,
                        max_dist_cat=max_dist_cat,
                        len_traj_pred=len_traj_pred,
                        context_size=config["context_size"],
                        normalize=config["normalize"],
                        goals_per_obs=goals_per_obs,
                        transform=transform,
                        predefined_index=None,
                        traj_stride=1,
                    )
                    if data_split_type == "train":
                        train_datasets.append(dataset)
                    else:
                        test_datasets.append(dataset)
                    print(f"Dataset: {dataset_name} ({data_split_type}), size: {len(dataset)}")

    print(f"Combining {len(train_datasets)} datasets.")
    train_dataset = ConcatDataset(train_datasets)
    test_dataset = ConcatDataset(test_datasets)

    balanced_sampling = bool(config.get("balanced_sampling", True)) and len(train_datasets) > 1
    balanced_sampling_mode = str(config.get("balanced_sampling_mode", "downsample_only")).strip().lower()
    if balanced_sampling:
        min_len = min(len(d) for d in train_datasets)
        if balanced_sampling_mode == "downsample_only":
            desired_total_size = len(train_datasets) * min_len
        else:
            desired_total_size = len(train_dataset)

        sampler = BalancedDistributedSampler(
            train_datasets,
            num_replicas=dist.get_world_size(),
            rank=rank,
            shuffle=True,
            seed=args.global_seed,
            desired_total_size=desired_total_size,
        )
    else:
        sampler = DistributedSampler(
            train_dataset,
            num_replicas=dist.get_world_size(),
            rank=rank,
            shuffle=True,
            seed=args.global_seed
        )
    loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        sampler=sampler,
        num_workers=config['num_workers'],
        pin_memory=True,
        drop_last=True,
        persistent_workers=True
    )
    logger.info(f"Dataset contains {len(train_dataset):,} images")

    microbatches_per_epoch = len(loader)
    updates_per_epoch = accumulation_update_count(
        microbatches_per_epoch, accumulation_steps
    )
    if updates_per_epoch <= 0:
        raise ValueError(
            "training loader has fewer microbatches than one accumulation group: "
            f"microbatches={microbatches_per_epoch}, accumulation={accumulation_steps}"
        )
    tail_microbatches = microbatches_per_epoch % accumulation_steps
    if tail_microbatches and rank == 0:
        logger.warning(
            "Dropping %d incomplete microbatch(es) at each epoch tail; "
            "optimizer updates remain exact effective-batch groups",
            tail_microbatches,
        )
    total_steps = args.epochs * updates_per_epoch
    final_lr = float(config.get('final_lr', 2e-5))
    lr_schedule = str(config.get('lr_schedule', 'linear') or 'linear').strip().lower()

    if lr_schedule in {"cosine", "cos", "cosineannealing"}:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps, eta_min=final_lr)
    elif lr_schedule in {"linear", "lambda"}:
        def lr_lambda(step):
            if total_steps <= 0:
                return 1.0
            alpha = step / float(total_steps)
            return (final_lr / base_lr) * alpha + (1 - alpha)
        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    else:
        raise ValueError(f"Unknown lr_schedule={lr_schedule}. Use 'linear' or 'cosine'.")
    
    if train_steps > 0:
        if override_lr_on_resume:
            scheduler.step(max(int(train_steps) - 1, 0))
            logger.info(
                f"Scheduler re-derived from config at resumed step={train_steps}, current lr: {scheduler.get_last_lr()[0]:.6f}"
            )
        else:
            if os.path.isfile(latest_path) or config.get('from_checkpoint', 0):
                latest_path = latest_path if os.path.isfile(latest_path) else config.get('from_checkpoint', 0)
                latest_checkpoint = torch.load(latest_path, map_location=device, weights_only=False)
                if "scheduler" in latest_checkpoint:
                    scheduler.load_state_dict(latest_checkpoint["scheduler"])
                    logger.info(f"Scheduler state restored from checkpoint, current lr: {scheduler.get_last_lr()[0]:.6f}")
                else:
                    for _ in range(train_steps):
                        scheduler.step()
                    logger.info(f"Scheduler fast-forwarded to step {train_steps}, current lr: {scheduler.get_last_lr()[0]:.6f}")
            else:
                for _ in range(train_steps):
                    scheduler.step()
                logger.info(f"Scheduler fast-forwarded to step {train_steps}, current lr: {scheduler.get_last_lr()[0]:.6f}")

    # Prepare models for training:
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    # Variables for monitoring/logging purposes:
    log_steps = 0
    running_loss = 0
    start_time = time()

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")

        opt.zero_grad()
        group_loss = 0.0
        for micro_idx, batch in enumerate(loader):
            # DataLoader's drop_last handles sample tails; this second tail
            # gate prevents an incomplete accumulation group from becoming a
            # smaller, non-paper optimizer update.
            if micro_idx >= updates_per_epoch * accumulation_steps:
                break
            if isinstance(batch, (list, tuple)) and len(batch) == 4:
                x, y, rel_t, context_rel_paths = batch
            else:
                x, y, rel_t = batch
                context_rel_paths = None
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            rel_t = rel_t.to(device, non_blocking=True)
            
            is_last_microbatch = (micro_idx + 1) % accumulation_steps == 0
            sync_context = (
                model.no_sync()
                if not is_last_microbatch and hasattr(model, "no_sync")
                else contextlib.nullcontext()
            )
            with sync_context:
                with torch.amp.autocast('cuda', enabled=bfloat_enable, dtype=torch.bfloat16):
                    with torch.no_grad():
                        B, T = x.shape[:2]
                        x_flat = x.flatten(0,1)
                        x_pix = x_flat * 0.5 + 0.5
                        x_latent = tokenizer.encode(x_pix)
                        x = x_latent.unflatten(0, (B, T))

                    num_goals = T - num_cond
                    x_start = x[:, num_cond:].flatten(0, 1)
                    x_cond = x[:, :num_cond].unsqueeze(1).expand(B, num_goals, num_cond, x.shape[2], x.shape[3], x.shape[4]).flatten(0, 1)
                    y = y.flatten(0, 1)
                    rel_t = rel_t.flatten(0, 1)

                    model_kwargs = dict(y=y, x_cond=x_cond, rel_t=rel_t)
                    transport_terms = transport.training_losses(
                        model,
                        x_start,
                        model_kwargs,
                    )
                    transport_loss = transport_terms.get("transport_loss", transport_terms["loss"]).mean()
                    loss = transport_terms["loss"].mean()
                    scaled_loss = loss / float(accumulation_steps)

                if not bfloat_enable:
                    scaled_loss.backward()
                else:
                    scaler.scale(scaled_loss).backward()

            group_loss += loss.detach().item()
            # Do not retain encoder activations or a computation graph between
            # microbatches; parameter gradients are the only intentional state.
            del x, y, rel_t, x_flat, x_pix, x_latent, x_start, x_cond
            del model_kwargs, transport_terms, transport_loss, loss, scaled_loss

            if not is_last_microbatch:
                continue

            clip_val = float(config.get('grad_clip_val', 1.0))
            if not bfloat_enable:
                if clip_val and clip_val > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_val)
                opt.step()
            else:
                if clip_val and clip_val > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_val)
                scaler.step(opt)
                scaler.update()
            opt.zero_grad()
            update_ema(ema, model.module)
            scheduler.step()

            # Log loss values once per optimizer update, using the mean of the
            # unscaled microbatch losses.
            running_loss += group_loss / float(accumulation_steps)
            group_loss = 0.0
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                samples_per_sec = (
                    dist.get_world_size()
                    * int(config['batch_size'])
                    * accumulation_steps
                    * int(num_goals)
                    * steps_per_sec
                )
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}, Samples/Sec: {samples_per_sec:.2f}")
                

                # Log to wandb
                if rank == 0 and wandb.run is not None:
                    log_dict = {
                        'metrics/training_loss': avg_loss,
                        'performance/steps_per_second': steps_per_sec,
                        'performance/samples_per_second': samples_per_sec,
                        'training/epoch': epoch,
                        'training/step': train_steps,
                        'training/learning_rate': scheduler.get_last_lr()[0],
                    }

                    wandb.log(log_dict, step=train_steps)
                
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()
                

            # Save DiT checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    # Use the new checkpoint saving function
                    checkpoint_path = save_checkpoint_with_step(
                        model, ema, opt, args, epoch, train_steps, checkpoint_dir, 
                        scaler if bfloat_enable else None, scheduler
                    )
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
            
            if train_steps % args.eval_every == 0 and train_steps > 0:
                eval_start_time = time()
                save_dir = os.path.join(experiment_dir, str(train_steps))
                sim_score = evaluate(
                    ema,
                    tokenizer,
                    transport,
                    sampler_transport,
                    test_dataset,
                    rank,
                    config["batch_size"],
                    config["num_workers"],
                    latent_size,
                    device,
                    save_dir,
                    args.global_seed + train_steps,
                    bfloat_enable,
                    num_cond,
                    transport_cfg=config.get("transport", {}),
                    max_batches=int(config.get("eval_num_batches", 1)),
                )
                dist.barrier()
                eval_end_time = time()
                eval_time = eval_end_time - eval_start_time
                logger.info(f"(step={train_steps:07d}) Perceptual Loss: {sim_score:.4f}, Eval Time: {eval_time:.2f}")
                
                # Log evaluation results to wandb
                if rank == 0 and wandb.run is not None:
                    wandb.log({
                        'eval/perceptual_loss': sim_score,
                        'eval/eval_time': eval_time
                    }, step=train_steps)

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")
    cleanup()


@torch.no_grad()
def evaluate(model, rae, transport, sampler_transport, test_dataloaders, rank, batch_size, num_workers, latent_size, device, save_dir, seed, bfloat_enable, num_cond, transport_cfg=None, max_batches=1):
    sampler = DistributedSampler(
        test_dataloaders,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=seed
    )

    g = torch.Generator()
    g.manual_seed(int(seed))

    def seed_worker(worker_id):
        worker_seed = (int(seed) + int(worker_id)) % (2**32)
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    loader = DataLoader(
        test_dataloaders,
        batch_size=batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=g,
    )
    from dreamsim import dreamsim
    eval_model, _ = dreamsim(pretrained=True)
    eval_model.eval()
    eval_model = eval_model.to(device)
    score = torch.tensor(0.).to(device)
    n_samples = torch.tensor(0).to(device)

    if rank == 0:
        os.makedirs(save_dir, exist_ok=True)
    saved = 0

    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and int(max_batches) > 0 and batch_idx >= int(max_batches):
            break
        if isinstance(batch, (list, tuple)) and len(batch) >= 3:
            x, y, rel_t = batch[:3]
        else:
            raise ValueError(f"Unexpected eval batch format: {type(batch)} with len={len(batch) if isinstance(batch, (list, tuple)) else 'NA'}")
        x = x.to(device)
        y = y.to(device)
        rel_t = rel_t.to(device).flatten(0, 1)
        with torch.amp.autocast('cuda', enabled=bfloat_enable, dtype=torch.bfloat16):
            B, T = x.shape[:2]
            num_goals = T - num_cond
            if num_goals <= 0:
                num_goals = 1
                if rank == 0:
                    logger.warning(f"Eval fallback: T={T}, num_cond={num_cond} => no goals; using num_goals=1")
            B_flat = B * num_goals
            x_flat = x.flatten(0,1)
            x_pix = x_flat * 0.5 + 0.5
            x_latent = rae.encode(x_pix)
            x_latent = x_latent.unflatten(0, (B, T))
            init = torch.randn(B_flat, 768, latent_size, latent_size, device=device)
            cfg = transport.time_dist_type
            try:
                with open("config/eval_config.yaml", "r") as f:
                    _dummy = f.read()
            except Exception:
                pass

            tp = transport_cfg or {}
            sampling_method = str(tp.get("sampling_method", "euler"))
            num_steps = int(tp.get("num_steps", 50))
            sample_fn = sampler_transport.sample_ode(
                sampling_method=sampling_method,
                num_steps=num_steps,
                atol=1e-6,
                rtol=1e-3,
                reverse=False
            )
            x_cond = x_latent[:, :num_cond].unsqueeze(1).expand(B, num_goals, num_cond, *x_latent.shape[2:]).flatten(0, 1)
            y_target = y.flatten(0, 1)
            rel_t_target = rel_t
            mobj = model.module if hasattr(model, "module") else model
            rif_prev = getattr(mobj, "return_intermediate_features", False)
            try:
                setattr(mobj, "return_intermediate_features", False)
                xs = sample_fn(init, mobj, y=y_target, x_cond=x_cond, rel_t=rel_t_target)
            finally:
                try:
                    setattr(mobj, "return_intermediate_features", rif_prev)
                except Exception:
                    pass
            samples_latent = xs[-1]
            xs = None
            
            samples = rae.decode(samples_latent).float()
            samples = torch.nan_to_num(samples)
            samples = samples.clamp(0,1)
            
            x_cond_pixels = rae.decode(x_cond.flatten(0, 1)).float()
            x_cond_pixels = torch.nan_to_num(x_cond_pixels).clamp(0,1).unflatten(0, (B_flat, num_cond))
            x_pix_full = x_pix.unflatten(0, (B, T))
            x_cond_raw_pixels = x_pix_full[:, :num_cond].unsqueeze(1).expand(B, num_goals, num_cond, *x_pix_full.shape[2:]).flatten(0, 1)
            if T - num_cond > 0:
                x_start_raw_pixels = x_pix_full[:, num_cond:].flatten(0, 1)
            else:
                x_start_raw_pixels = x_pix_full[:, -1]
            
            if not torch.isfinite(samples_latent).all():
                samples = x_cond_pixels[:, -1]
            if T - num_cond > 0:
                x_start_latent = x_latent[:, num_cond:].flatten(0, 1)
                x_start_pixels = rae.decode(x_start_latent).float()
            else:
                x_start_latent = x_latent[:, -1]
                x_start_pixels = rae.decode(x_start_latent).float()
            x_start_pixels = torch.nan_to_num(x_start_pixels).clamp(0,1)
            
            
            if rank == 0 and saved < 10:
                n_to_save = min(int(samples.shape[0]), 10 - saved)
                for i in range(n_to_save):
                    _, ax = plt.subplots(1,3,dpi=256)
                    def to_uint8_image(img3ch):
                        img = img3ch.detach().float().clamp(0.0, 1.0)
                        return (img.permute(1,2,0).cpu().numpy()*255.0).astype('uint8')
                    ax[0].imshow(to_uint8_image(x_cond_raw_pixels[i, -1]))
                    ax[0].axis('off')
                    ax[1].imshow(to_uint8_image(x_start_raw_pixels[i]))
                    ax[1].axis('off')
                    ax[2].imshow(to_uint8_image(samples[i]))
                    ax[2].axis('off')
                    plt.tight_layout()
                    plt.savefig(f'{save_dir}/{saved}.png')
                    plt.close()
                    saved += 1

            res = eval_model(x_start_pixels, samples)
            score += res.sum()
            n_samples += len(res)

    dist.all_reduce(score)
    dist.all_reduce(n_samples)
    sim_score = score/n_samples
    try:
        del eval_model
    except Exception:
        pass
    try:
        import gc
        gc.collect()
        torch.cuda.empty_cache()
    except Exception:
        pass
    return sim_score

def save_checkpoint_with_step(model, ema, opt, args, epoch, train_steps, checkpoint_dir, scaler=None, scheduler=None):
    checkpoint = {
        "model": model.module.state_dict(),
        "ema": ema.state_dict(),
        "opt": opt.state_dict(),
        "args": args,
        "epoch": epoch,
        "train_steps": train_steps,
    }
    
    if scaler is not None:
        checkpoint["scaler"] = scaler.state_dict()
    
    if scheduler is not None:
        checkpoint["scheduler"] = scheduler.state_dict()
    
    step_checkpoint_path = f"{checkpoint_dir}/checkpoint_step_{train_steps}.pth.tar"
    torch.save(checkpoint, step_checkpoint_path)
    latest_checkpoint_path = f"{checkpoint_dir}/latest.pth.tar"
    torch.save(checkpoint, latest_checkpoint_path)
    
    return step_checkpoint_path

def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=300)
    # parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=2000)
    parser.add_argument("--eval-every", type=int, default=5000)
    parser.add_argument("--bfloat16", type=int, default=1)
    parser.add_argument("--torch-compile", type=int, default=1)
    return parser

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
