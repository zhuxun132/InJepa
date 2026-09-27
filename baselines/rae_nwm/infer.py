# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from distributed import init_distributed
import torch
import os
import sys
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RAE_SRC = os.path.join(PROJECT_ROOT, "RAE", "src")
if RAE_SRC not in sys.path:
    sys.path.append(RAE_SRC)
import math


import yaml
import argparse
import os
import numpy as np
from RAE.src.utils.model_utils import instantiate_from_config
from RAE.src.utils.train_utils import parse_configs
from RAE.src.stage1.rae import RAE
from RAE.src.stage2.transport.transport import Transport, ModelType, PathType, WeightType, Sampler
import misc
import distributed as dist
from models import CDiT_models
from datasets import EvalDataset
from PIL import Image
import torch.nn.functional as F
import math
from time import time


def save_image(output_file, img):
    img = img.detach().cpu().float()
    img = torch.nan_to_num(img).clamp(0, 1)
    img_np = (img.permute(1, 2, 0).numpy() * 255).astype('uint8')
    image = Image.fromarray(img_np, mode='RGB')
    image.save(output_file)
    
    
def get_dataset_eval(config, dataset_name, eval_type, predefined_index=True):
    data_config = config["eval_datasets"][dataset_name]    
    if predefined_index:
        predefined_index = f"data_splits/{dataset_name}/test/{eval_type}.pkl"
    else:
        predefined_index=None

    
    dataset = EvalDataset(
                data_folder=data_config["data_folder"],
                data_split_folder=data_config["test"],
                dataset_name=dataset_name,
                image_size=config["image_size"],
                min_dist_cat=config["eval_distance"]["eval_min_dist_cat"],
                max_dist_cat=config["eval_distance"]["eval_max_dist_cat"],
                len_traj_pred=config["eval_len_traj_pred"],
                traj_stride=config["traj_stride"], 
                context_size=config["eval_context_size"],
                normalize=config["normalize"],
                transform=misc.transform,
                goals_per_obs=4,
                predefined_index=predefined_index,
                traj_names='traj_names.txt'
            )
    
    return dataset

@torch.no_grad()
def model_forward_wrapper(all_models, curr_obs, curr_delta, num_timesteps, latent_size, device, num_cond, sampling_method, num_steps, num_goals=1, rel_t=None, input_is_latent=False):
    model, transport, sampler, rae = all_models
    x = curr_obs.to(device)
    y = curr_delta.to(device)

    with torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16):
        B, T = x.shape[:2]

        if rel_t is None:
            rel_t = (torch.ones(B)* (1. / 128.)).to(device)
            rel_t *= num_timesteps

        if not input_is_latent:
            x = x.flatten(0, 1)
            x_pix = x * 0.5 + 0.5
            x = rae.encode(x_pix).unflatten(0, (B, T))
        else:
            pass

        x_cond = x[:, :num_cond].unsqueeze(1).expand(B, num_goals, num_cond, x.shape[2], x.shape[3], x.shape[4]).flatten(0, 1)
        
        z = torch.randn(B*num_goals, rae.latent_dim, latent_size, latent_size, device=device)
        y = y.flatten(0, 1)
        model_kwargs = dict(y=y, x_cond=x_cond, rel_t=rel_t)      
        
        sample_fn = sampler.sample_ode(
            sampling_method=sampling_method,
            num_steps=num_steps,
            atol=1e-6,
            rtol=1e-3,
            reverse=False
        )
        xs = sample_fn(z, model.module if hasattr(model, "module") else model, y=y, x_cond=x_cond, rel_t=rel_t)
        
        samples_latent = xs[-1]

        samples = rae.decode(samples_latent).float()
        samples = torch.nan_to_num(samples)

        return torch.clamp(samples, 0.0, 1.0), samples_latent

def generate_rollout(args, output_dir, rollout_fps, idxs, all_models, obs_image, gt_image, delta, num_cond, device, perf=None, profile_state=None):
    rollout_stride = args.input_fps // rollout_fps
    gt_image = gt_image[:, rollout_stride-1::rollout_stride]

    delta = delta.unflatten(1, (-1, rollout_stride))  # (B, T', stride, D)
    B, T_group, S, D = delta.shape
    if D >= 3:
        x = torch.zeros(B, T_group, device=delta.device, dtype=delta.dtype)
        y = torch.zeros_like(x)
        th = torch.zeros_like(x)
        for s in range(S):
            dx = delta[:, :, s, 0]
            dy = delta[:, :, s, 1]
            dth = delta[:, :, s, 2]
            c = torch.cos(th)
            si = torch.sin(th)
            x = x + c * dx - si * dy
            y = y + si * dx + c * dy
            th = th + dth
        two_pi = 2.0 * math.pi
        th = th - two_pi * torch.floor((th + math.pi) / two_pi)
        delta_se2 = torch.stack([x, y, th], dim=-1)
        if D > 3:
            delta_rest = delta[:, :, :, 3:].sum(dim=2)
            delta = torch.cat([delta_se2, delta_rest], dim=-1)
        else:
            delta = delta_se2
    else:
        delta = delta.sum(2)

    if not args.gt:
        _, _, _, rae = all_models
        with torch.no_grad():
            init_obs = obs_image.to(device).flatten(0, 1) * 0.5 + 0.5
            with torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16):
                curr_latents = rae.encode(init_obs)
            curr_latents = curr_latents.unflatten(0, (obs_image.shape[0], obs_image.shape[1]))

    for i in range(gt_image.shape[1]):
        curr_delta = delta[:, i:i+1].to(device)
        
        if args.gt:
            x_pred_pixels = (gt_image[:, i].clone().to(device) * 0.5 + 0.5).clamp(0, 1)
        else:
            bs = curr_latents.shape[0]
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.synchronize()
            t0 = time()
            
            input_latents = curr_latents[:, -num_cond:]

            x_pred_pixels, x_pred_latent = model_forward_wrapper(
                all_models,
                input_latents,
                curr_delta,
                rollout_stride,
                args.latent_size,
                num_cond=num_cond,
                sampling_method=args.sampling_method,
                num_steps=args.num_steps,
                num_goals=1,
                device=device,
                input_is_latent=True,
            )
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time()
            elapsed = t1 - t0
            
            mem_mb = 0.0
            if torch.cuda.is_available():
                mem_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
            print(f"[Infer] rollout_step={i} batch_size={bs} elapsed={elapsed:.3f}s, peak_mem={mem_mb:.1f}MB")
            if perf is not None:
                p = perf['rollout']
                p['time_total'] += elapsed
                p['step_count'] += 1
                p['mem_total'] += mem_mb
                p['sample_total'] += bs
                p['mem_peak_overall_mb'] = max(p['mem_peak_overall_mb'], mem_mb)

            if profile_state is not None:
                profile_state['seen'] = int(profile_state.get('seen', 0)) + 1

            x_pred_latent = x_pred_latent.unsqueeze(1)
            curr_latents = torch.cat((curr_latents, x_pred_latent), dim=1)
            curr_latents = curr_latents[:, -num_cond:]

        visualize_preds(output_dir, idxs, i, x_pred_pixels)

def generate_time(args, output_dir, idxs, all_models, obs_image, gt_output, delta, secs, num_cond, device, perf=None, profile_state=None):
    eval_timesteps = [sec*args.input_fps for sec in secs]

    base_obs = obs_image
    target_h, target_w = base_obs.shape[-2], base_obs.shape[-1]

    for sec, timestep in zip(secs, eval_timesteps):
        delta_seq = delta[:, :timestep].to(device)
        B, T, D = delta_seq.shape
        if D >= 3:
            x = torch.zeros(B, device=device, dtype=delta_seq.dtype)
            y = torch.zeros_like(x)
            th = torch.zeros_like(x)
            for s in range(T):
                dx = delta_seq[:, s, 0]
                dy = delta_seq[:, s, 1]
                dth = delta_seq[:, s, 2]
                c = torch.cos(th)
                si = torch.sin(th)
                x = x + c * dx - si * dy
                y = y + si * dx + c * dy
                th = th + dth
            two_pi = 2.0 * math.pi
            th = th - two_pi * torch.floor((th + math.pi) / two_pi)
            delta_se2 = torch.stack([x, y, th], dim=-1)
            if D > 3:
                delta_rest = delta_seq[:, :, 3:].sum(dim=1)
                delta_comp = torch.cat([delta_se2, delta_rest], dim=-1)
            else:
                delta_comp = delta_se2
        else:
            delta_comp = delta_seq.sum(dim=1)
        curr_delta = delta_comp.unsqueeze(1)
        if args.gt:
            x_pred_pixels = (gt_output[:, timestep-1].clone().to(device) * 0.5 + 0.5).clamp(0, 1)
            x_pred_latent = None
        else:
            bs = base_obs.shape[0]
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.synchronize()

            t0 = time()
            x_pred_pixels, x_pred_latent = model_forward_wrapper(
                all_models,
                base_obs,
                curr_delta,
                timestep,
                args.latent_size,
                num_cond=num_cond,
                sampling_method=args.sampling_method,
                num_steps=args.num_steps,
                num_goals=1,
                device=device,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time()
            elapsed = t1 - t0
            mem_mb = 0.0
            if torch.cuda.is_available():
                mem_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
            print(f"[Infer] time_sec={sec} batch_size={bs} elapsed={elapsed:.3f}s, peak_mem={mem_mb:.1f}MB")
            if perf is not None:
                p = perf['time']
                p['time_total'] += elapsed
                p['step_count'] += 1
                p['mem_total'] += mem_mb
                p['sample_total'] += bs
                p['mem_peak_overall_mb'] = max(p['mem_peak_overall_mb'], mem_mb)

            if profile_state is not None:
                profile_state['seen'] = int(profile_state.get('seen', 0)) + 1

        if x_pred_pixels.shape[-2:] != (target_h, target_w):
            x_pred_pixels = F.interpolate(
                x_pred_pixels,
                size=(target_h, target_w),
                mode='bicubic',
                align_corners=False,
            )

        visualize_preds(output_dir, idxs, sec, x_pred_pixels)

def visualize_preds(output_dir, idxs, sec, x_pred_pixels):
    flat_idxs = idxs.view(-1)
    for batch_idx, sample_idx in enumerate(flat_idxs):
        sample_idx = int(sample_idx.item())
        sample_folder = os.path.join(output_dir, f'id_{sample_idx}')
        os.makedirs(sample_folder, exist_ok=True)
        image_file = os.path.join(sample_folder, f'{int(sec)}.png')
        save_image(image_file, x_pred_pixels[batch_idx])




def _is_sample_complete(output_dir, sample_idx, expected_secs):
    sample_folder = os.path.join(output_dir, f'id_{int(sample_idx)}')
    if not os.path.isdir(sample_folder):
        return False
    if expected_secs is None or len(expected_secs) == 0:
        return False
    last_sec = expected_secs[-1]
    return os.path.isfile(os.path.join(sample_folder, f'{int(last_sec)}.png'))


def _filter_batch_by_existing(output_dir, idxs, obs_image, gt_image, delta, expected_secs):
    idxs_1d = idxs.view(-1)
    keep = []
    for i in range(int(idxs_1d.shape[0])):
        sid = int(idxs_1d[i].item())
        if not _is_sample_complete(output_dir, sid, expected_secs):
            keep.append(i)
    if len(keep) == 0:
        return None

    keep_t_cpu = torch.as_tensor(keep, device='cpu', dtype=torch.long)

    def _sel0(x):
        return x.index_select(0, keep_t_cpu.to(device=x.device))

    return (
        idxs_1d.index_select(0, keep_t_cpu.to(device=idxs_1d.device)),
        _sel0(obs_image),
        _sel0(gt_image),
        _sel0(delta),
    )

@torch.no_grad
def main(args):
    _, _, device, _ = init_distributed()
    device = torch.device(device)
    num_tasks = dist.get_world_size()
    global_rank = dist.get_rank()
    exp_eval = args.exp

    # model & config setup
    if args.gt:
        args.save_output_dir = os.path.join(args.output_dir, 'gt')
    else:
        save_name = None
        if args.checkpoint_path:
            ckpt_dir = os.path.dirname(os.path.abspath(args.checkpoint_path))
            run_dir = os.path.dirname(ckpt_dir)
            save_name = os.path.basename(run_dir)
        if not save_name:
            save_name = os.path.basename(exp_eval).split('.')[0]
        save_name = f"{save_name}_{args.sampling_method}"
        args.save_output_dir = os.path.join(args.output_dir, save_name)

    os.makedirs(args.save_output_dir, exist_ok=True)

    with open("config/eval_config.yaml", "r") as f:
        default_config = yaml.safe_load(f)
    config = default_config

    with open(exp_eval, "r") as f:
        user_config = yaml.safe_load(f)
    config.update(user_config)

    if args.eval_min_dist_cat is not None or args.eval_max_dist_cat is not None:
        config.setdefault('eval_distance', {})
        if args.eval_min_dist_cat is not None:
            config['eval_distance']['eval_min_dist_cat'] = int(args.eval_min_dist_cat)
        if args.eval_max_dist_cat is not None:
            config['eval_distance']['eval_max_dist_cat'] = int(args.eval_max_dist_cat)

    if args.eval_len_traj_pred is not None:
        config['eval_len_traj_pred'] = int(args.eval_len_traj_pred)

    latent_size = config['image_size'] // 14
    args.latent_size = latent_size

    num_cond = config['context_size']
    model_lst = (None, None, None)
    if not args.gt:
        config_path = config.get('config_path', 'RAE/configs/stage1/pretrained/DINOv2-B.yaml')
        rae_config, *_ = parse_configs(config_path)
        rae: RAE = instantiate_from_config(rae_config).to(device).eval()
        model_kwargs = {
            'context_size': num_cond,
            'input_size': latent_size,
            'in_channels': rae.latent_dim,
            'learn_sigma': bool(config.get('learn_sigma', False)),
            'head_width': config.get('head_width', rae.latent_dim),
            'head_depth': int(config.get('head_depth', 2)),
            'head_num_heads': int(config.get('head_num_heads', 16)),
        }
        try:
            model = CDiT_models[config['model']](**model_kwargs)
        except TypeError:
            model_kwargs.pop('head_width', None)
            model_kwargs.pop('head_depth', None)
            model_kwargs.pop('head_num_heads', None)
            model = CDiT_models[config['model']](**model_kwargs)
        
        checkpoint_path = None
        
        if args.checkpoint_path:
            checkpoint_path = args.checkpoint_path
            
        elif 'checkpoint_path' in config and config['checkpoint_path']:
            checkpoint_path = config['checkpoint_path']
            
        elif 'checkpoint_name' in config and config['checkpoint_name']:
            checkpoint_path = f'{config["results_dir"]}/{config["run_name"]}/checkpoints/{config["checkpoint_name"]}.pth.tar'
            
        else:
            checkpoint_path = f'{config["results_dir"]}/{config["run_name"]}/checkpoints/{args.ckp}.pth.tar'
            
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
            
        ckp = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        _state = {k.replace('_orig_mod.', ''): v for k, v in ckp.get("ema", {}).items()}
        incompatible = model.load_state_dict(_state, strict=False)
        try:
            mk = getattr(incompatible, 'missing_keys', [])
            uk = getattr(incompatible, 'unexpected_keys', [])
            print(f"[Infer] State load: missing={len(mk)} unexpected={len(uk)}")
        except Exception:
            pass

        
        model.eval()
        model.to(device)
        model = torch.compile(model)

        tp = config.get("transport", {}) if isinstance(config, dict) else {}
        latent_size = int(config["image_size"]) // 14
        shift_dim = int(rae.latent_dim) * int(latent_size) * int(latent_size)
        shift_base = float(tp.get("time_dist_shift_base", 4096))
        time_dist_shift = math.sqrt(float(shift_dim) / float(shift_base))
        if 'time_dist_shift' in tp and tp.get('time_dist_shift') is not None:
            time_dist_shift = float(tp.get('time_dist_shift'))
        if bool(tp.get('time_dist_shift_disable', False)):
            time_dist_shift = 1.0
        print(f"[Infer] Using time_dist_shift={time_dist_shift:.4f} = sqrt({shift_dim}/{shift_base}).")

        transport = Transport(
            model_type=getattr(ModelType, str(tp.get('model_type', 'velocity')).upper()),
            path_type=getattr(PathType, str(tp.get('path_type', 'linear')).upper()),
            loss_type=getattr(WeightType, str(tp.get('loss_type', 'velocity')).upper()),
            time_dist_type=str(tp.get('time_dist_type', 'uniform')),
            time_dist_shift=time_dist_shift,
            train_eps=1e-3,
            sample_eps=1e-3,
        )
        sampler = Sampler(transport)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device], find_unused_parameters=False)
        model_lst = (model, transport, sampler, rae)

    # Loading Datasets
    dataset_names = args.datasets.split(',')
    datasets = {}

    for dataset_name in dataset_names:
        dataset_val = get_dataset_eval(config, dataset_name, args.eval_type, predefined_index=False)

        if len(dataset_val) % num_tasks != 0:
            print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                    'This will slightly alter validation results as extra duplicate entries are added to achieve '
                    'equal num of samples per-process.')
        sampler_val = torch.utils.data.DistributedSampler(
            dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)

        curr_data_loader = torch.utils.data.DataLoader(
                            dataset_val, sampler=sampler_val,
                            batch_size=args.batch_size,
                            num_workers=args.num_workers,
                            pin_memory=True,
                            drop_last=False
                        )
        datasets[dataset_name] = curr_data_loader

    print_freq = 1
    header = 'Evaluation: '
    metric_logger = dist.MetricLogger(delimiter="  ")
    perf = {
        'rollout': {'time_total': 0.0, 'step_count': 0, 'mem_total': 0.0, 'mem_peak_overall_mb': 0.0, 'sample_total': 0},
        'time': {'time_total': 0.0, 'step_count': 0, 'mem_total': 0.0, 'mem_peak_overall_mb': 0.0, 'sample_total': 0},
    }
    profile_state = {'done': False, 'seen': 0, 'target': 10}


    max_ids = int(getattr(args, 'max_ids', 0) or 0)
    per_rank_remaining = None
    if max_ids > 0:
        per_rank_max = int(math.ceil(float(max_ids) / float(max(1, num_tasks))))
        per_rank_remaining = per_rank_max

    for dataset_name in dataset_names:
        dataset_save_output_dir = os.path.join(args.save_output_dir, dataset_name)
        os.makedirs(dataset_save_output_dir, exist_ok=True)
        curr_data_loader = datasets[dataset_name]

        for data_iter_step, batch in enumerate(metric_logger.log_every(curr_data_loader, print_freq, header)):
            if per_rank_remaining is not None and int(per_rank_remaining) <= 0:
                break
            with torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16):
                if isinstance(batch, (list, tuple)) and len(batch) == 4:
                    idxs, obs_image, gt_image, delta = batch
                else:
                    idxs, obs_image, gt_image, delta = batch

                idxs = idxs.view(-1)

                if per_rank_remaining is not None:
                    take = min(int(idxs.shape[0]), int(per_rank_remaining))
                    if take <= 0:
                        break
                    idxs = idxs[:take]
                    obs_image = obs_image[:take]
                    gt_image = gt_image[:take]
                    delta = delta[:take]

                obs_image = obs_image[:, -num_cond:].to(device)
                gt_image = gt_image.to(device)
                num_cond = config["context_size"]

                if args.eval_type == 'rollout':
                    for rollout_fps in args.rollout_fps_values:
                        curr_rollout_output_dir = os.path.join(dataset_save_output_dir, f'rollout_{rollout_fps}fps')
                        os.makedirs(curr_rollout_output_dir, exist_ok=True)

                        rollout_stride = args.input_fps // int(rollout_fps)
                        start = int(rollout_stride) - 1
                        gt_len = int(gt_image.shape[1])
                        if gt_len <= start:
                            continue
                        num_frames = ((gt_len - start - 1) // int(rollout_stride)) + 1
                        expected_frames = list(range(int(num_frames)))

                        filtered = _filter_batch_by_existing(
                            curr_rollout_output_dir,
                            idxs,
                            obs_image,
                            gt_image,
                            delta,
                            expected_frames,
                        )
                        if filtered is None:
                            continue
                        idxs_f, obs_f, gt_f, delta_f = filtered

                        generate_rollout(args, curr_rollout_output_dir, rollout_fps, idxs_f, model_lst, obs_f, gt_f, delta_f, num_cond, device, perf, profile_state=profile_state)

                        if per_rank_remaining is not None:
                            per_rank_remaining -= int(idxs_f.shape[0])
                            if int(per_rank_remaining) <= 0:
                                break

                        p = perf['rollout']
                        if p['step_count'] > 0:
                            avg_bt = p['time_total'] / p['step_count']
                            avg_st = p['time_total'] / max(1, p['sample_total'])
                            avg_mem = p['mem_total'] / p['step_count']
                            print(f"[Infer RunningAvg] dataset={dataset_name} mode=rollout steps={p['step_count']} avg_batch_time={avg_bt:.3f}s avg_sample_time={avg_st:.3f}s avg_peak_mem={avg_mem:.1f}MB overall_peak_mem={p['mem_peak_overall_mb']:.1f}MB")

                elif args.eval_type == 'time':
                    secs = np.array([2**i for i in range(0, args.num_sec_eval)])
                    curr_time_output_dir = os.path.join(dataset_save_output_dir, 'time')
                    os.makedirs(curr_time_output_dir, exist_ok=True)

                    filtered = _filter_batch_by_existing(
                        curr_time_output_dir,
                        idxs,
                        obs_image,
                        gt_image,
                        delta,
                        secs.tolist(),
                    )
                    if filtered is None:
                        continue
                    idxs_f, obs_f, gt_f, delta_f = filtered

                    generate_time(args, curr_time_output_dir, idxs_f, model_lst, obs_f, gt_f, delta_f, secs, num_cond, device, perf, profile_state=profile_state)

                    if per_rank_remaining is not None:
                        per_rank_remaining -= int(idxs_f.shape[0])

                    p = perf['time']
                    if p['step_count'] > 0:
                        avg_bt = p['time_total'] / p['step_count']
                        avg_st = p['time_total'] / max(1, p['sample_total'])
                        avg_mem = p['mem_total'] / p['step_count']
                        print(f"[Infer RunningAvg] dataset={dataset_name} mode=time steps={p['step_count']} avg_batch_time={avg_bt:.3f}s avg_sample_time={avg_st:.3f}s avg_peak_mem={avg_mem:.1f}MB overall_peak_mem={p['mem_peak_overall_mb']:.1f}MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--output_dir", type=str, default=None, help="output directory")
    parser.add_argument("--exp", type=str, default=None, help="experiment name")
    parser.add_argument("--ckp", type=str, default='0100000')
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Full checkpoint file path; if provided, --ckp is ignored")
    parser.add_argument("--eval_min_dist_cat", type=int, default=None, help="Override eval_distance.eval_min_dist_cat for this run")
    parser.add_argument("--eval_max_dist_cat", type=int, default=None, help="Override eval_distance.eval_max_dist_cat for this run")
    parser.add_argument("--eval_len_traj_pred", type=int, default=None, help="Override eval_len_traj_pred for this run")
    parser.add_argument("--num_sec_eval", type=int, default=5)
    parser.add_argument("--input_fps", type=int, default=4)
    parser.add_argument("--datasets", type=str, default=None, help="dataset name")
    parser.add_argument("--num_workers", type=int, default=8, help="num workers")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--eval_type", type=str, default=None, help="type of evaluation has to be either 'time' or 'rollout'")
    # Rollout Evaluation Args
    parser.add_argument("--rollout_fps_values", type=str, default='1,4', help="")
    parser.add_argument("--gt", type=int, default=0, help="set to 1 to produce ground truth evaluation set")
    parser.add_argument("--sampling_method", type=str, default="euler")
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--max_ids", type=int, default=0, help="Max number of sample IDs to generate/process (0 means no limit)")
    args = parser.parse_args()
    
    args.rollout_fps_values = [int(fps) for fps in args.rollout_fps_values.split(',')]
    
    main(args)
