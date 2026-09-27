# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.nn.functional as F

import argparse
import yaml
import os
import json
import numpy as np
import lpips
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

import sys
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RAE_SRC = os.path.join(PROJECT_ROOT, "RAE", "src")
if RAE_SRC not in sys.path:
    sys.path.append(RAE_SRC)
from RAE.src.utils.model_utils import instantiate_from_config
from RAE.src.utils.train_utils import parse_configs
from RAE.src.stage1.rae import RAE
from RAE.src.stage2.transport.transport import Transport, ModelType, PathType, WeightType, Sampler
import math
from datetime import datetime

### evo evaluation library ###
from evo.core.trajectory import PoseTrajectory3D
from evo.core import sync, metrics
import evo.main_ape as main_ape
import evo.main_rpe as main_rpe
from evo.core.metrics import PoseRelation

from diffusion import create_diffusion
from datasets import TrajectoryEvalDataset
from infer import model_forward_wrapper
from misc import calculate_delta_yaw, get_action_torch, save_planning_pred, log_viz_single, transform, unnormalize_data
from evaluate import save_metric_to_disk
import distributed as dist
from models import CDiT_models


with open("config/data_config.yaml", "r") as f:
    data_config = yaml.safe_load(f)

with open("config/data_hyperparams_plan.yaml", "r") as f:
    data_hyperparams = yaml.safe_load(f)

ACTION_STATS_TORCH = {}
for key in data_config['action_stats']:
    ACTION_STATS_TORCH[key] = torch.tensor(data_config['action_stats'][key])


def plot_images_with_losses(preds, losses, save_path="predictions_with_losses.png"):
    preds = (preds + 1) / 2
    ncol = int(preds.size(0) ** 0.5)
    nrow = preds.size(0) // ncol
    if ncol * nrow < preds.size(0):
        nrow += 1
    grid = vutils.make_grid(preds, nrow=ncol, padding=2)
    np_grid = (grid.clamp(0, 1).mul(255).permute(1, 2, 0).byte().cpu().numpy())
    H, W = np_grid.shape[:2]
    img = Image.fromarray(np_grid, mode="RGB")
    draw = ImageDraw.Draw(img)
    img_height = H // nrow
    img_width = W // ncol
    for idx, loss in enumerate(losses):
        row = idx // ncol
        col = idx % ncol
        x = col * img_width + img_width // 2
        y = row * img_height + 15
        text = "GT Goal" if idx == 0 else f"ID: {idx - 1}  Loss: {loss:.2f}"
        draw.text((x, y), text, fill=(255, 255, 255))
    img.save(save_path)

def plot_batch_final(init_imgs, pred_imgs, goal_imgs, idxs, losses, save_path="final_plan.png"):
    imgs_for_plotting = torch.cat([init_imgs, pred_imgs, goal_imgs])
    imgs_for_plotting = (imgs_for_plotting + 1) / 2
    ncol = init_imgs.shape[0]
    grid = vutils.make_grid(imgs_for_plotting, nrow=ncol, padding=2)
    np_grid = (grid.clamp(0, 1).mul(255).permute(1, 2, 0).byte().cpu().numpy())
    H, W = np_grid.shape[:2]
    img = Image.fromarray(np_grid, mode="RGB")
    draw = ImageDraw.Draw(img)
    img_height = H // 3
    img_width = W // ncol
    for i in range(ncol):
        x = i * img_width + img_width // 2
        y_pred = img_height
        draw.text((x, y_pred + 15), f"ID: {int(idxs[i].item())} Loss: {losses[i]:.2f}", fill=(255, 255, 255))
    img.save(save_path)

def get_dataset_eval(config, dataset_name, predefined_index=True, subset_items=None, subset_seed=0):
    data_config = config["eval_datasets"][dataset_name]
    
    if subset_items is not None and subset_items > 0:
        predefined_index = None
    elif predefined_index:
        nav_idx = f"data_splits/{dataset_name}/test/navigation_eval.pkl"
        predefined_index = nav_idx if os.path.isfile(nav_idx) else None
    else:
        predefined_index = None

    split_root = data_config["test"]
    traj_file = "rollout_traj_names.txt"
    if not os.path.exists(os.path.join(split_root, traj_file)):
        traj_file = "traj_names.txt"

    dataset = TrajectoryEvalDataset(
                data_folder=data_config["data_folder"],
                data_split_folder=split_root,
                dataset_name=dataset_name,
                image_size=config["image_size"],
                min_dist_cat=config["trajectory_eval_distance"]["min_dist_cat"],
                max_dist_cat=config["trajectory_eval_distance"]["max_dist_cat"],
                len_traj_pred=config["trajectory_eval_len_traj_pred"],
                traj_stride=config["traj_stride"], 
                context_size=config["trajectory_eval_context_size"],
                normalize=config["normalize"],
                transform=transform,
                predefined_index=predefined_index,
                traj_names=traj_file
            )
    
    if subset_items is not None and subset_items > 0:
        import random
        print(f"Sampling {subset_items} items from {len(dataset)} items with seed {subset_seed}...")
        rng = random.Random(subset_seed)
        rng.shuffle(dataset.index_to_data)
        dataset.index_to_data = dataset.index_to_data[:min(subset_items, len(dataset.index_to_data))]
        print(f"Dataset size after subset: {len(dataset)}")
        
    return dataset

class WM_Planning_Evaluator:
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.exp = args.exp
        self.prior_mix = float(getattr(self.args, "prior_mix", 0.0) or 0.0)
        self.backtrack_allow = float(getattr(self.args, "backtrack_allow", 0.0) or 0.0)
        self.prior_beta = float(getattr(self.args, "prior_beta", 0.8) or 0.8)

        _, _, gpu, _ = dist.init_distributed()
        self.local_rank = int(gpu)
        self.device = torch.device(f"cuda:{self.local_rank}")

        num_tasks = dist.get_world_size()
        global_rank = dist.get_rank()

        self.exp_eval = self.exp

        with open("config/eval_config.yaml", "r") as f:
            default_config = yaml.safe_load(f)
        self.config = default_config

        with open(self.exp_eval, "r") as f:
            user_config = yaml.safe_load(f)
        self.config.update(user_config)

        args_score_type = str(getattr(self.args, "score_type", "dino") or "dino").strip().lower()
        cfg_score_type = str(self.config.get("planning_score_type", args_score_type) or args_score_type).strip().lower()
        if cfg_score_type not in {"dino", "lpips"}:
            raise ValueError(f"Unsupported score_type: {cfg_score_type}")
        self.score_type = cfg_score_type
        self.base_score_type = self.score_type

        self.planning_score_type = self.score_type
        self.planning_base_score_type = self.base_score_type

        cfg_traj_sampler = str(self.config.get("planning_traj_sampler", "") or "").strip().lower()
        if not cfg_traj_sampler:
            cfg_traj_sampler = "curve"
        arg_traj_sampler = str(getattr(self.args, "traj_sampler", "") or "").strip().lower()
        if not arg_traj_sampler:
            self.args.traj_sampler = cfg_traj_sampler
        else:
            self.args.traj_sampler = arg_traj_sampler

        self.get_eval_name()

        latent_size = self.config['image_size'] // 14
        self.latent_size = latent_size
        self.num_cond = self.config['eval_context_size']
        
        exp_name = os.path.basename(self.args.exp).split('.')[0]
        run_tag = str(getattr(self.args, 'run_tag', '') or '').strip()
        if not run_tag:
            run_tag = datetime.now().strftime('%Y%m%d_%H%M%S')
        traj_sampler = str(getattr(self.args, "traj_sampler", "curve") or "curve").strip().lower()
        if traj_sampler not in {"curve", "line"}:
            traj_sampler = "curve"
        self.run_name = f"{traj_sampler}_{self.eval_name}_{run_tag}"
        self.args.save_output_dir = os.path.join(args.output_dir, exp_name, self.run_name)
        os.makedirs(self.args.save_output_dir, exist_ok=True)

        if dist.is_main_process():
            with open(os.path.join(self.args.save_output_dir, "traj_sampler.txt"), "w") as f:
                f.write(str(getattr(self.args, "traj_sampler", "")))
            with open(os.path.join(self.args.save_output_dir, "run_meta.json"), "w") as f:
                json.dump(
                    {
                        "traj_sampler": str(getattr(self.args, "traj_sampler", "")),
                        "score_type": str(getattr(self, "score_type", "")),
                        "eval_name": str(getattr(self, "eval_name", "")),
                        "run_name": str(getattr(self, "run_name", "")),
                        "exp": str(getattr(self.args, "exp", "")),
                        "cmd": " ".join(sys.argv),
                    },
                    f,
                    indent=2,
                )
            with open(os.path.join(self.args.save_output_dir, "config_used.yaml"), "w") as f:
                yaml.safe_dump(self.config, f, sort_keys=False)
                
        # Loading Datasets
        self.dataset_names = self.args.datasets.split(',')
        self.datasets = {}
        for dataset_name in self.dataset_names:
            dataset_val = get_dataset_eval(self.config, dataset_name, predefined_index=True, subset_items=getattr(self.args, 'subset_items', None), subset_seed=getattr(self.args, 'subset_seed', 0))
            
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                        'This will slightly alter validation results as extra duplicate entries are added to achieve '
                        'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)

            curr_data_loader = torch.utils.data.DataLoader(
                                            dataset_val, sampler=sampler_val,
                                            batch_size=self.args.batch_size,
                                            num_workers=self.args.num_workers,
                                            pin_memory=True,
                                            drop_last=False
                                        )
            self.datasets[dataset_name] = curr_data_loader
        
        # Loading Model
        print("loading")
        config_path = self.config.get('config_path', 'RAE/configs/stage1/pretrained/DINOv2-B.yaml')
        rae_config, *_ = parse_configs(config_path)
        self.rae = instantiate_from_config(rae_config).to(self.device).eval()
        model = CDiT_models[self.config['model']](
            context_size=self.num_cond,
            input_size=latent_size,
            in_channels=self.rae.latent_dim,
            head_width=self.config.get('head_width', self.rae.latent_dim),
            head_depth=int(self.config.get('head_depth', 2)),
            head_num_heads=int(self.config.get('head_num_heads', 16)),
            learn_sigma=bool(self.config.get('learn_sigma', False)),
        )
        checkpoint_path = None
        if hasattr(args, 'checkpoint_path') and args.checkpoint_path:
            checkpoint_path = args.checkpoint_path
            print(f"Using checkpoint from --checkpoint_path: {checkpoint_path}")
        elif self.config.get('checkpoint_path'):
            checkpoint_path = self.config['checkpoint_path']
            print(f"Using checkpoint_path from config: {checkpoint_path}")
        elif self.config.get('checkpoint_name'):
            checkpoint_path = f'{self.config["results_dir"]}/{self.config["run_name"]}/checkpoints/{self.config["checkpoint_name"]}.pth.tar'
            print(f"Using checkpoint_name from config: {self.config['checkpoint_name']}")
        else:
            checkpoint_path = f'{self.config["results_dir"]}/{self.config["run_name"]}/checkpoints/{args.ckp}.pth.tar'
            print(f"Using fallback --ckp: {args.ckp}")
        ckp = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        _state = {k.replace('_orig_mod.', ''): v for k, v in ckp.get("ema", {}).items()}
        model.load_state_dict(_state, strict=False)
        model.eval()
        model.to(self.device)
        self.model = torch.compile(model)
        self.model = torch.nn.parallel.DistributedDataParallel(
            self.model,
            device_ids=[self.local_rank],
            output_device=self.local_rank,
            find_unused_parameters=False,
        )
        tp = self.config.get('transport', {})
        shift_dim = self.rae.latent_dim * self.latent_size * self.latent_size
        shift_base = float(tp.get('time_dist_shift_base', 4096))
        time_dist_shift = math.sqrt(shift_dim / shift_base)
        self.transport = Transport(
            model_type=getattr(ModelType, str(tp.get('model_type', 'velocity')).upper()),
            path_type=getattr(PathType, str(tp.get('path_type', 'linear')).upper()),
            loss_type=getattr(WeightType, str(tp.get('loss_type', 'velocity')).upper()),
            time_dist_type=str(tp.get('time_dist_type', 'uniform')),
            time_dist_shift=time_dist_shift,
            train_eps=1e-3,
            sample_eps=1e-3
        )
        self.sampler = Sampler(self.transport)
        self.model_without_ddp = self.model.module
         
        self.loss_fn = None
        if self.base_score_type == "lpips":
            self.loss_fn = lpips.LPIPS(net="alex").to(self.device)

        self.mode = 'cem' # assume CEM for planning
        self.num_samples = self.args.num_samples
        self.topk = self.args.topk
        self.opt_steps = self.args.opt_steps
        self.num_repeat_eval = self.args.num_repeat_eval
        self.action_dim = 3

    def init_mu_sigma(self, obs_0, num_control_points):
        n_evals = obs_0.shape[0]
        mu = torch.zeros(n_evals, num_control_points, self.action_dim)
        sigma = torch.ones(n_evals, num_control_points, self.action_dim)
        
        base_mu = torch.tensor(data_hyperparams[self.args.datasets]['mu'])
        base_var = torch.tensor(data_hyperparams[self.args.datasets]['var_scale'])
        
        mu[:] = base_mu
        sigma[:] = base_var
        
        return mu, sigma

    def init_mu_sigma_line(self, obs_0):
        n_evals = obs_0.shape[0]
        mu = torch.zeros(n_evals, self.action_dim)
        sigma = torch.ones(n_evals, self.action_dim)
        base_mu = torch.tensor(data_hyperparams[self.args.datasets]['mu'])
        base_var = torch.tensor(data_hyperparams[self.args.datasets]['var_scale'])
        mu[:] = base_mu
        sigma[:] = base_var
        return mu, sigma

    def apply_motion_prior(self, dxy: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        beta = float(self.prior_beta)
        backtrack_allow = float(self.backtrack_allow)

        out = dxy.clone()
        if out.numel() == 0:
            return out

        out = out.to(dtype=torch.float32)
        n, t, _ = out.shape

        if t <= 1:
            return out.to(dtype=dxy.dtype)

        d0 = out[:, 0]
        n0 = d0.norm(dim=-1, keepdim=True)
        d1 = out[:, 1]
        n1 = d1.norm(dim=-1, keepdim=True)
        dir0 = torch.where(n0 > float(eps), d0 / (n0 + float(eps)), d1 / (n1 + float(eps)))
        dir0 = torch.nan_to_num(dir0)

        dir = dir0
        for ti in range(1, t):
            prev = out[:, ti - 1]
            prev_n = prev.norm(dim=-1, keepdim=True)
            prev_dir = prev / (prev_n + float(eps))
            prev_dir = torch.nan_to_num(prev_dir)

            dir_new = beta * dir + (1.0 - beta) * prev_dir
            dir_n = dir_new.norm(dim=-1, keepdim=True)
            dir = torch.where(dir_n > float(eps), dir_new / (dir_n + float(eps)), dir)
            dir = torch.nan_to_num(dir)

            step = out[:, ti]
            proj0 = (step * dir).sum(dim=-1, keepdim=True)
            perp = step - proj0 * dir
            proj = proj0.clamp(min=-backtrack_allow)
            out[:, ti] = perp + proj * dir

        return out.to(dtype=dxy.dtype)

    def encode_to_latent(self, img_norm):
        img_norm = img_norm.to(self.device)
        with torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16):
            x_pix = img_norm * 0.5 + 0.5
            return self.rae.encode(x_pix)

    def dino_latent_loss_flat(self, pred_latent, goal_latent):
        pred = pred_latent.float().flatten(2).transpose(1, 2)
        goal = goal_latent.float().flatten(2).transpose(1, 2)
        pred = F.normalize(pred, dim=-1)
        goal = F.normalize(goal, dim=-1)
        dist = 1.0 - (pred * goal).sum(dim=-1)
        return dist.mean(dim=-1)

    def dino_latent_loss(self, pred_latent, goal_latent):
        pred = F.normalize(pred_latent.float(), dim=1)
        goal = F.normalize(goal_latent.float(), dim=1)
        dist = 1.0 - (pred * goal).sum(dim=1)
        return dist.mean(dim=(1, 2))

    def decode_latent_to_img(self, latents, target_h, target_w):
        with torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16):
            samples = self.rae.decode(latents).float()
        samples = torch.nan_to_num(samples)
        if samples.shape[-2:] != (target_h, target_w):
            samples = F.interpolate(
                samples,
                size=(target_h, target_w),
                mode="bicubic",
                align_corners=False,
            )
        samples = torch.clamp(samples, 0.0, 1.0)
        return samples * 2.0 - 1.0

    def autoregressive_rollout_latent(self, obs_image, deltas, rollout_stride, return_sequence=True):
        deltas = deltas.unflatten(1, (-1, rollout_stride)).sum(2)
        device = self.device

        obs_image = obs_image.to(device)
        B, T = obs_image.shape[:2]

        with torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16):
            x_pix = obs_image * 0.5 + 0.5
            x_latent = self.rae.encode(x_pix.flatten(0, 1)).unflatten(0, (B, T))
        curr_latents = x_latent

        pred_latents = []
        for i in range(deltas.shape[1]):
            curr_delta = deltas[:, i:i+1].to(device)
            B_step = curr_delta.shape[0]
            y = curr_delta.flatten(0, 1)

            with torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16):
                rel_t = torch.ones(B_step, device=device) * (1.0 / 128.0)
                rel_t *= rollout_stride

                x_cond = curr_latents[:, :self.num_cond]
                x_cond = x_cond.unsqueeze(1).expand(
                    B_step,
                    1,
                    self.num_cond,
                    x_cond.shape[2],
                    x_cond.shape[3],
                    x_cond.shape[4],
                ).flatten(0, 1)

                z = torch.randn(
                    B_step,
                    self.rae.latent_dim,
                    self.latent_size,
                    self.latent_size,
                    device=device,
                )

                sample_fn = self.sampler.sample_ode(
                    sampling_method="euler",
                    num_steps=50,
                    atol=1e-6,
                    rtol=1e-3,
                    reverse=False,
                )
                xs = sample_fn(
                    z,
                    self.model_without_ddp,
                    y=y,
                    x_cond=x_cond,
                    rel_t=rel_t,
                )
                samples_latent = xs[-1]

            pred_latents.append(samples_latent.unsqueeze(1))
            curr_latents = torch.cat((curr_latents[:, 1:], samples_latent.unsqueeze(1)), dim=1)

        pred_latents = torch.cat(pred_latents, 1)
        if return_sequence:
            return pred_latents
        return pred_latents[:, -1]
        
    def generate_actions(self, dataset_save_output_dir, dataset_name, idxs, obs_image, goal_image, gt_actions, len_traj_pred):
        idx_string = "_".join(map(str, idxs.flatten().int().tolist()))

        plot_root = dataset_save_output_dir if dataset_save_output_dir is not None else self.args.save_output_dir
        image_plot_dir = None
        if self.args.plot or self.args.save_preds:
            image_plot_dir = os.path.join(plot_root, 'plots')
            os.makedirs(image_plot_dir, exist_ok=True)

        n_evals = obs_image.shape[0]
        goal_img = goal_image.squeeze(1) if goal_image.dim() == 5 else goal_image
        goal_latent_all = self.encode_to_latent(goal_img)

        traj_sampler = str(getattr(self.args, "traj_sampler", "curve") or "curve").strip().lower()
        if traj_sampler not in {"curve", "line"}:
            traj_sampler = "curve"

        num_control_points = 3
        if traj_sampler == "curve":
            mu, sigma = self.init_mu_sigma(obs_image, num_control_points)
        else:
            mu, sigma = self.init_mu_sigma_line(obs_image)
        mu, sigma = mu.to(self.device), sigma.to(self.device)

        for i in range(self.opt_steps):
            losses = []
            for traj in range(n_evals):
                traj_id = int(idxs.flatten()[traj].item())
                
                if traj_sampler == "curve":
                    sample = (
                        torch.randn(self.num_samples, num_control_points, self.action_dim, device=self.device)
                        * sigma[traj]
                        + mu[traj]
                    )
                    traj_ctrl_pos = sample[..., :2].permute(0, 2, 1)
                    deltas_pos = F.interpolate(
                        traj_ctrl_pos,
                        size=len_traj_pred,
                        mode='linear',
                        align_corners=True
                    )
                    deltas = deltas_pos.permute(0, 2, 1)
                else:
                    sample = (torch.randn(self.num_samples, self.action_dim).to(self.device) * sigma[traj] + mu[traj])
                    single_delta = sample[:, :2]
                    deltas = single_delta.unsqueeze(1).repeat(1, len_traj_pred, 1)

                if self.prior_mix != 0.0:
                    dxy2 = self.apply_motion_prior(deltas)
                    deltas = (1.0 - float(self.prior_mix)) * deltas + float(self.prior_mix) * dxy2.to(deltas)

                unnorm_deltas = unnormalize_data(deltas, ACTION_STATS_TORCH)
                delta_yaw = calculate_delta_yaw(unnorm_deltas)
                deltas = torch.cat((deltas, delta_yaw.to(deltas.device)), dim=-1)

                if traj_sampler == "curve":
                    terminal_bias = sample[:, -1, 2]
                else:
                    terminal_bias = sample[:, 2]
                deltas[:, -1, -1] += terminal_bias * np.pi

                cur_obs_image = obs_image[traj].unsqueeze(0).repeat(self.num_samples, 1, 1, 1, 1) 
                cur_goal_image = goal_image[traj].unsqueeze(0).repeat(self.args.num_samples, 1, 1, 1, 1).squeeze(1)

                # WM is stochastic, so we can repeat the evaluation of each trajectory and average to reduce variance
                if self.num_repeat_eval * self.num_samples > 120:
                    cur_losses = []
                    pred_latents_vis = None

                    for r in range(self.num_repeat_eval):
                        pred_latents = self.autoregressive_rollout_latent(
                            cur_obs_image, deltas, self.args.rollout_stride, return_sequence=False
                        )
                        if pred_latents_vis is None:
                            pred_latents_vis = pred_latents

                        base_loss = None
                        preds_r = None
                        if self.base_score_type == "dino":
                            goal_latents = goal_latent_all[traj].unsqueeze(0).repeat(self.num_samples, 1, 1, 1)
                            base_loss = self.dino_latent_loss(pred_latents, goal_latents).flatten(0)
                        else:
                            preds_r = self.decode_latent_to_img(pred_latents, cur_obs_image.shape[-2], cur_obs_image.shape[-1])
                            goal_imgs = goal_img[traj].unsqueeze(0).repeat(self.num_samples, 1, 1, 1)
                            base_loss = self.loss_fn(preds_r.to(self.device), goal_imgs.to(self.device)).flatten(0)

                        cur_losses.append(base_loss)

                    loss = torch.stack(cur_losses).mean(dim=0)
                    if self.args.plot:
                        preds = self.decode_latent_to_img(pred_latents_vis, cur_obs_image.shape[-2], cur_obs_image.shape[-1])
                else:
                    expanded_deltas = deltas.repeat(self.num_repeat_eval, 1, 1)
                    expanded_obs_image = cur_obs_image.repeat(self.num_repeat_eval, 1, 1, 1, 1)

                    pred_latents = self.autoregressive_rollout_latent(
                        expanded_obs_image, expanded_deltas, self.args.rollout_stride, return_sequence=False
                    )


                    if self.base_score_type == "dino":
                        expanded_goal_latents = goal_latent_all[traj].unsqueeze(0).repeat(
                            self.num_repeat_eval * self.num_samples, 1, 1, 1
                        )
                        base_loss = self.dino_latent_loss(pred_latents, expanded_goal_latents).flatten(0)
                    else:
                        preds_all = self.decode_latent_to_img(pred_latents, cur_obs_image.shape[-2], cur_obs_image.shape[-1])
                        expanded_goal_imgs = goal_img[traj].unsqueeze(0).repeat(
                            self.num_repeat_eval * self.num_samples, 1, 1, 1
                        )
                        base_loss = self.loss_fn(preds_all.to(self.device), expanded_goal_imgs.to(self.device)).flatten(0)

                    base_loss = base_loss.view(self.num_repeat_eval, -1)
                    base_loss = base_loss.mean(dim=0)

                    pred_latents = pred_latents[:self.args.num_samples]
                    if self.base_score_type == "lpips":
                        preds = preds_all[: self.args.num_samples]
                    elif self.args.plot:
                        preds = self.decode_latent_to_img(pred_latents, cur_obs_image.shape[-2], cur_obs_image.shape[-1])

                    loss = base_loss

                sorted_idx = torch.argsort(loss)
                topk_idx = sorted_idx[:self.topk]
                
                topk_samples = sample[topk_idx]
                losses.append(loss[topk_idx[0]].item())
                mu[traj] = topk_samples.mean(dim=0)
                sigma[traj] = topk_samples.std(dim=0)

                if self.args.plot:
                    plot_topn = getattr(self.args, "plot_topn", self.args.num_samples)
                    if plot_topn is None or plot_topn <= 0:
                        plot_idx = sorted_idx
                    else:
                        plot_idx = sorted_idx[: min(int(plot_topn), sorted_idx.numel())]
                    self.visualize_trajectories(dataset_name, gt_actions, image_plot_dir, i, traj, traj_id, deltas, cur_obs_image, cur_goal_image, preds, loss, plot_idx, topk_idx)
        
        if traj_sampler == "curve":
            final_ctrl_pos = mu[..., :2].permute(0, 2, 1)
            final_deltas_pos = F.interpolate(
                final_ctrl_pos,
                size=len_traj_pred,
                mode='linear',
                align_corners=True
            ).permute(0, 2, 1)
            deltas = final_deltas_pos
        else:
            deltas = mu[:, :2].unsqueeze(1).repeat(1, len_traj_pred, 1)

        if self.prior_mix != 0.0:
            dxy2 = self.apply_motion_prior(deltas)
            deltas = (1.0 - float(self.prior_mix)) * deltas + float(self.prior_mix) * dxy2.to(deltas)

        unnorm_deltas = unnormalize_data(deltas, ACTION_STATS_TORCH)
        delta_yaw = calculate_delta_yaw(unnorm_deltas)
        deltas = torch.cat((deltas, delta_yaw.to(deltas.device)), dim=-1)

        if traj_sampler == "curve":
            terminal_bias = mu[:, -1, 2]
        else:
            terminal_bias = mu[:, 2]
        deltas[:, -1, -1] += terminal_bias * np.pi

        pred_latents = self.autoregressive_rollout_latent(obs_image, deltas, self.args.rollout_stride, return_sequence=False)
        preds = self.decode_latent_to_img(pred_latents, obs_image.shape[-2], obs_image.shape[-1])

        if self.base_score_type == "dino":
            base_loss = self.dino_latent_loss(pred_latents, goal_latent_all).flatten(0)
        else:
            base_loss = self.loss_fn(preds.to(self.device), goal_img.to(self.device)).flatten(0)

        loss = base_loss

        if self.args.save_preds:
            save_planning_pred(dataset_save_output_dir, n_evals, idxs, obs_image, goal_image, preds, deltas, loss, gt_actions)
        
        if self.args.plot:
            img_name = os.path.join(image_plot_dir, f'FINAL_{idx_string}.png')
            plot_batch_final(obs_image[:, -1].to(self.device), preds, goal_image.squeeze(1).to(self.device), idxs, losses, save_path=img_name)

        pred_actions = get_action_torch(deltas[:, :, :2], ACTION_STATS_TORCH)
        pred_yaw = deltas[:, :, -1].sum(1)
        return pred_actions, pred_yaw

    def visualize_trajectories(self, dataset_name, gt_actions, image_plot_dir, i, traj, traj_id, deltas, cur_obs_image, cur_goal_image, preds, loss, plot_idx, topk_idx):
        preds = preds[plot_idx]
        deltas = deltas[plot_idx]
        loss = loss[plot_idx]

        img_for_plotting = torch.cat([cur_goal_image[0:1].to(self.device), preds])
        loss_for_plotting = torch.cat((torch.tensor([0], device=self.device), loss))
        img_name = os.path.join(image_plot_dir, f'idx{traj_id}_iter{i}.png')
        plot_images_with_losses(img_for_plotting, loss_for_plotting, save_path=img_name)
        plot_name = os.path.join(image_plot_dir, f'idx{traj_id}_iter{i}_trajs.png')
        highlight_idx = torch.tensor([0], device=self.device)
        log_viz_single(
                        dataset_name, 
                        cur_obs_image[0], 
                        cur_goal_image[0], 
                        preds, 
                        deltas, 
                        loss, 
                        highlight_idx, 
                        gt_actions[traj], 
                        ACTION_STATS_TORCH, 
                        plan_iter=i, 
                        output_dir=plot_name
                    )
    
    def autoregressive_rollout(self, obs_image, deltas, rollout_stride, return_sequence=True):
        device = self.device
        obs_image = obs_image.to(device)
        B = obs_image.shape[0]
        target_h, target_w = obs_image.shape[-2], obs_image.shape[-1]

        pred_latents = self.autoregressive_rollout_latent(obs_image, deltas, rollout_stride, return_sequence=True)
        T_rollout = pred_latents.shape[1]

        if return_sequence:
            decoded = self.decode_latent_to_img(pred_latents.flatten(0, 1), target_h, target_w)
            return decoded.unflatten(0, (B, T_rollout))

        return self.decode_latent_to_img(pred_latents[:, -1], target_h, target_w)
    
    def get_eval_name(self):
        traj_sampler = str(getattr(self.args, "traj_sampler", "curve") or "curve").strip().lower()
        if traj_sampler == "line":
            self.eval_name = f'{self.score_type}_line_CEM_N{self.args.num_samples}_K{self.args.topk}_RS{self.args.rollout_stride}_rep{self.args.num_repeat_eval}_OPT{self.args.opt_steps}'
        else:
            self.eval_name = f'{self.score_type}_CEM_N{self.args.num_samples}_K{self.args.topk}_RS{self.args.rollout_stride}_rep{self.args.num_repeat_eval}_OPT{self.args.opt_steps}'
        
    def actions_to_traj(self, actions):
        actions = actions.detach().to(device="cpu", dtype=torch.float64)

        positions_xyz = torch.zeros((actions.shape[0], 3), dtype=torch.float64)
        positions_xyz[:, :2] = actions

        orientations_quat_wxyz = torch.zeros((actions.shape[0], 4), dtype=torch.float64)
        orientations_quat_wxyz[:, -1] = 1

        timestamps = torch.arange(actions.shape[0], dtype=torch.float64)
        traj = PoseTrajectory3D(
            positions_xyz=positions_xyz,
            orientations_quat_wxyz=orientations_quat_wxyz,
            timestamps=timestamps,
        )
        return traj
    
    @torch.no_grad
    def evaluate(self):
        
        for dataset_name in self.dataset_names:
            metric_logger = dist.MetricLogger(delimiter="  ")
            header = 'Test:'
            eval_save_output_dir = None
            
            if self.args.save_preds:
                dataset_save_output_dir = os.path.join(self.args.save_output_dir, dataset_name)
                os.makedirs(dataset_save_output_dir, exist_ok=True)
                eval_save_output_dir = os.path.join(dataset_save_output_dir, self.eval_name)
                os.makedirs(eval_save_output_dir, exist_ok=True)
            
            curr_data_loader = self.datasets[dataset_name]
            for (idxs, obs_image, goal_image, gt_actions, goal_pos) in metric_logger.log_every(curr_data_loader, 1, header):
                obs_image = obs_image[:, -self.num_cond:]
                with torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16):
                    pred_actions, pred_yaw = self.generate_actions(eval_save_output_dir, dataset_name, idxs, obs_image, goal_image, gt_actions, self.config["trajectory_eval_len_traj_pred"])
                for i in range(len(obs_image)):
                    pred_traj_i = self.actions_to_traj(pred_actions[i, :, :2])
                    gt_traj_i = self.actions_to_traj(gt_actions[i, :, :2])
                    
                    ate, rpe_trans, _ = self.eval_metrics(gt_traj_i, pred_traj_i)

                    pred_final_pos = pred_actions[i, -1, :2].to('cpu') # (2,)
                    pred_final_yaw = pred_yaw[i].to('cpu') # 
                    goal_final_pos = goal_pos[i, 0, :2] # (2,)
                    goal_final_yaw = goal_pos[i, 0, -1] # (B,)
                    pos_diff_norm = torch.norm(pred_final_pos - goal_final_pos)
                    yaw_diff = pred_final_yaw - goal_final_yaw   # 
                    yaw_diff_norm = torch.atan2(torch.sin(yaw_diff), torch.cos(yaw_diff)).abs()
                    
                    metric_logger.meters['{}_ate'.format(dataset_name)].update(ate, n=1)
                    metric_logger.meters['{}_rpe_trans'.format(dataset_name)].update(rpe_trans, n=1)
                    metric_logger.meters['{}_pos_diff_norm'.format(dataset_name)].update(pos_diff_norm, n=1)   
                    metric_logger.meters['{}_yaw_diff_norm'.format(dataset_name)].update(yaw_diff_norm, n=1)   
            output_fn = os.path.join(self.args.save_output_dir, f'{dataset_name}_{self.run_name}.json')
            save_metric_to_disk(metric_logger, output_fn)

        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
            
    def eval_metrics(self, traj_ref, traj_pred):
        traj_ref, traj_pred = sync.associate_trajectories(traj_ref, traj_pred)
        
        result = main_ape.ape(traj_ref, traj_pred, est_name='traj',
            pose_relation=PoseRelation.translation_part, align=False, correct_scale=False)
        ate = result.stats['rmse']

        result = main_rpe.rpe(traj_ref, traj_pred, est_name='traj',
            pose_relation=PoseRelation.rotation_angle_deg, align=False, correct_scale=False,
            delta=1.0, delta_unit=metrics.Unit.frames, rel_delta_tol=0.1)
        rpe_rot = result.stats['rmse']

        result = main_rpe.rpe(traj_ref, traj_pred, est_name='traj',
            pose_relation=PoseRelation.translation_part, align=False, correct_scale=False,
            delta=1.0, delta_unit=metrics.Unit.frames, rel_delta_tol=0.1)
        rpe_trans = result.stats['rmse']

        return ate, rpe_trans, rpe_rot
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Default Args
    parser.add_argument("--exp", type=str, default=None, help="experiment name")
    parser.add_argument("--ckp", type=str, default='0100000', help="experiment name")

    parser.add_argument("--datasets", type=str, default=None, help="dataset name")
    parser.add_argument("--output_dir", type=str, default=None, help="output dir to save model predictions")
    parser.add_argument('--save_preds', action='store_true', default=False, help='whether to save prediction tensors or not')
    parser.add_argument("--num_workers", type=int, default=8, help="num workers")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="full path to checkpoint file, if provided overrides --ckp")
    parser.add_argument("--subset_items", type=int, default=0, help="number of evaluation samples (items) to run (0 means all)")
    parser.add_argument("--subset_seed", type=int, default=0, help="random seed for subset sampling")
    parser.add_argument("--run_tag", type=str, default="", help="optional run tag for unique output folder/name; default uses timestamp")
    
    # Planning Specific Args
    parser.add_argument("--num_samples", type=int, default=10, help="num nomad samples to predict")
    parser.add_argument("--rollout_stride", type=int, default=1, help="rollout stride")
    parser.add_argument("--topk", type=int, default=5, help="top k samples to take mean and var for CEM")
    parser.add_argument("--opt_steps", type=int, default=15, help="num iterations for CEM")
    parser.add_argument("--num_repeat_eval", type=int, default=1, help="number of evals for one action")
    parser.add_argument("--prior_mix", type=float, default=0.0)
    parser.add_argument("--backtrack_allow", type=float, default=0.0)
    parser.add_argument("--prior_beta", type=float, default=0.8)
    parser.add_argument(
        "--traj_sampler",
        type=str,
        default="",
        choices=["curve", "line"],
        help="trajectory sampling mode for CEM: curve (control points + interpolation) or line (constant delta); if omitted, uses planning_traj_sampler in the YAML config",
    )
    parser.add_argument('--plot', action='store_true', default=False)
    parser.add_argument("--plot_topn", type=int, default=10, help="only visualize top-n candidate trajectories (by loss) per CEM iteration; <=0 means all")
    parser.add_argument(
        "--score_type",
        type=str,
        default="dino",
        choices=["dino", "lpips"],
        help="scoring type for CEM selection: dino or lpips",
    )
    args = parser.parse_args()
    
    evaluator = WM_Planning_Evaluator(args)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    gpu_id = torch.cuda.current_device()  # Or args.gpu if explicitly set
    evaluator.evaluate()