from operator import truediv
import sys
import os

import os
import sys
import json
import pickle
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import torch.distributed as torch_dist
import distributed as dist
import lpips
try:
    from dreamsim import dreamsim
    _dreamsim_available = True
except Exception:
    _dreamsim_available = False
from torcheval.metrics import FrechetInceptionDistance


def _list_episode_dirs(root_dir: str):
    if not root_dir or not os.path.isdir(root_dir):
        return []
    out = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    out.sort()
    return out


# DINO Wrapper
class DINOMetric(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14').to(device)
        self.model.eval()

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, img0, img1):
        with torch.no_grad():
            img0 = img0.to(device=self.device, dtype=torch.float32)
            img1 = img1.to(device=self.device, dtype=torch.float32)

            img0 = F.interpolate(img0, size=(224, 224), mode="bicubic", align_corners=False,antialias=True)
            img1 = F.interpolate(img1, size=(224, 224), mode="bicubic", align_corners=False,antialias=True)

            mean = self.mean.to(device=img0.device, dtype=img0.dtype)
            std = self.std.to(device=img0.device, dtype=img0.dtype)
            img0 = (img0 - mean) / std
            img1 = (img1 - mean) / std

            feat0 = self.model.forward_features(img0)["x_norm_patchtokens"]
            feat1 = self.model.forward_features(img1)["x_norm_patchtokens"]

            feat0 = F.normalize(feat0.float(), dim=-1)
            feat1 = F.normalize(feat1.float(), dim=-1)
            dist = 1.0 - (feat0 * feat1).sum(dim=-1)
        return dist.mean()

# Depth Consistency Wrapper
class DepthConsistencyMetric(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.midas = torch.hub.load("intel-isl/MiDaS", "DPT_Large").to(device)
        self.midas.eval()
        
        self.register_buffer("mean", torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1))

    def forward(self, img0, img1):
        with torch.no_grad():
            img0 = img0.to(device=self.device, dtype=torch.float32)
            img1 = img1.to(device=self.device, dtype=torch.float32)

            input_size = (384, 384)
            in0 = F.interpolate(img0, size=input_size, mode="bicubic", align_corners=False,antialias=True)
            in1 = F.interpolate(img1, size=input_size, mode="bicubic", align_corners=False,antialias=True)

            mean = self.mean.to(device=in0.device, dtype=in0.dtype)
            std = self.std.to(device=in0.device, dtype=in0.dtype)
            in0 = (in0 - mean) / std
            in1 = (in1 - mean) / std

            depth0 = self.midas(in0)
            depth1 = self.midas(in1)

            def norm_depth(d):
                d = d.to(dtype=torch.float32)
                d_flat = d.flatten(1)
                d_min = d_flat.min(1)[0].view(-1, 1, 1)
                d_max = d_flat.max(1)[0].view(-1, 1, 1)
                return (d - d_min) / (d_max - d_min + 1e-6)

            depth0_n = norm_depth(depth0)
            depth1_n = norm_depth(depth1)

            return torch.abs(depth0_n - depth1_n).mean()


def get_loss_fn(loss_fn_type, secs, device):
    if loss_fn_type == 'lpips':
        general_lpips_loss_fn = lpips.LPIPS(net='alex').to(device)

        def loss_fn(img0: torch.Tensor, img1: torch.Tensor):
            img0 = img0.to(device=device, dtype=torch.float32)
            img1 = img1.to(device=device, dtype=torch.float32)
            img0 = img0 * 2.0 - 1.0
            img1 = img1 * 2.0 - 1.0
            dist = general_lpips_loss_fn(img0, img1)
            return dist.mean()

        return loss_fn

    elif loss_fn_type == 'dreamsim':
        if _dreamsim_available:
            dreamsim_loss_fn, preprocess = dreamsim(pretrained=True, device=device)
            to_pil = transforms.ToPILImage()
            def _preprocess_batch(x: torch.Tensor) -> torch.Tensor:
                try:
                    out = preprocess(x)
                    if isinstance(out, torch.Tensor):
                        if out.dim() == 3:
                            out = out.unsqueeze(0)
                        return out.to(device)
                except Exception:
                    pass
                out = []
                x_cpu = x.detach().cpu()
                for i in range(x_cpu.shape[0]):
                    out.append(preprocess(to_pil(x_cpu[i])).to(device))
                return torch.cat(out, dim=0)
            def loss_fn(img0: torch.Tensor, img1: torch.Tensor):
                img0 = img0.to(dtype=torch.float32)
                img1 = img1.to(dtype=torch.float32)
                all_img0 = _preprocess_batch(img0)
                all_img1 = _preprocess_batch(img1)
                dist = dreamsim_loss_fn(all_img0, all_img1)
                return dist.mean()
            return loss_fn
        else:
            def loss_fn(img0: torch.Tensor, img1: torch.Tensor):
                return torch.tensor(float('nan'), device=device)
            return loss_fn

    elif loss_fn_type == 'fid':
        fid_metrics = {}
        for sec in secs:
            fid_metrics[sec] = FrechetInceptionDistance(feature_dim=2048).to(device)
        return fid_metrics

    elif loss_fn_type == 'dino':
        dino_fn = DINOMetric(device)

        def loss_fn(img0: torch.Tensor, img1: torch.Tensor):
            return dino_fn(img0, img1)

        return loss_fn

    elif loss_fn_type == 'depth':
        depth_fn = DepthConsistencyMetric(device)

        def loss_fn(img0: torch.Tensor, img1: torch.Tensor):
            return depth_fn(img0, img1)

        return loss_fn

    else:
        raise NotImplementedError

def _sync_fid_metrics(fid_metrics, secs, device="cuda"):
    if not dist.is_dist_avail_and_initialized():
        return fid_metrics

    secs_list = [int(s) for s in list(secs)]

    serialized = pickle.dumps(fid_metrics)
    gathered = [None] * dist.get_world_size()

    torch_dist.barrier()
    torch_dist.all_gather_object(gathered, serialized)

    final = {s: FrechetInceptionDistance(feature_dim=2048).to(device) for s in secs_list}
    for ser in gathered:
        curr = pickle.loads(ser)
        for s in secs_list:
            final[s].merge_state([curr[s]])

    return final


def evaluate(args, dataset_name, eval_type, metric_logger, loss_fns, gt_dir, exp_dir, secs, rollout_fps):
    (lpips_fn, dreamsim_fn, fid_fn, dino_fn, depth_fn) = loss_fns
    use_fid = True
    if fid_fn is None:
        fid_fn = get_loss_fn('fid', secs, device='cuda')

    gt_eps = _list_episode_dirs(gt_dir)
    exp_eps = _list_episode_dirs(exp_dir)

    match_mode = str(getattr(args, "match_mode", "auto") or "auto").lower()
    if match_mode not in {"auto", "strict", "intersection"}:
        match_mode = "auto"

    gt_set = set(gt_eps)
    exp_set = set(exp_eps)

    if match_mode == "strict":
        if gt_set != exp_set:
            missing_in_pred = sorted(gt_set - exp_set)
            extra_in_pred = sorted(exp_set - gt_set)
            raise ValueError(
                f"GT/PRED episode sets differ. missing_in_pred={len(missing_in_pred)} extra_in_pred={len(extra_in_pred)}"
            )
        eps = gt_eps
    elif match_mode == "intersection":
        eps = sorted(gt_set & exp_set)
    else:
        if gt_set == exp_set:
            eps = gt_eps
        else:
            eps = sorted(gt_set & exp_set)

    max_trajs = int(getattr(args, "max_trajs", 0) or 0)
    if max_trajs > 0:
        eps = eps[:max_trajs]

    if len(eps) == 0:
        raise ValueError(f"No matched episodes under gt_dir={gt_dir} and exp_dir={exp_dir}")

    world_size = dist.get_world_size()
    global_rank = dist.get_rank()
    if world_size > 1:
        eps = eps[global_rank::world_size]
    if len(eps) == 0:
        return

    if eval_type == 'rollout':
        eval_name = f'rollout_{rollout_fps}fps'
        valid_eps = [ep for ep in eps if os.path.isdir(os.path.join(gt_dir, ep))]
        if len(valid_eps) == 0:
            return
        sample_ep_dir = os.path.join(gt_dir, valid_eps[0])
        frame_files = [f for f in os.listdir(sample_ep_dir) if f.endswith('.png')]
        max_idx = max(int(os.path.splitext(f)[0]) for f in frame_files)
        rollout_last_idx = getattr(args, "rollout_last_idx", None)
        if rollout_last_idx is not None:
            try:
                max_idx = min(int(max_idx), int(rollout_last_idx))
            except Exception:
                pass
        max_frames = max_idx + 1

        if rollout_fps == 1:
            image_idxs = np.arange(0, max_frames)
            secs = image_idxs.copy()
        else:
            first = np.arange(0, min(4, max_frames))
            every4 = np.arange(4, max_frames, 4)
            last = np.array([max_frames - 1])
            image_idxs = np.unique(np.concatenate([first, every4, last]))
            secs = image_idxs.copy()

        fid_fn = get_loss_fn('fid', secs, device='cuda')
    elif eval_type == 'time':
        eval_name = eval_type
        image_idxs = secs.copy()
        
    total_pairs = 0
    for batch_start in tqdm(
        range(0, len(eps), args.batch_size),
        total=(len(eps) + args.batch_size - 1) // args.batch_size,
        disable=not dist.is_main_process(),
    ):
        batch_eps = eps[batch_start:batch_start + args.batch_size]

        gt_batch, exp_batch = {}, {}
        for sec in secs:
            gt_batch[sec] = []
            exp_batch[sec] = []

        batch_pairs = 0
        for ep in batch_eps:
            gt_ep_dir = os.path.join(gt_dir, ep)
            exp_ep_dir = os.path.join(exp_dir, ep)

            if not os.path.isdir(gt_ep_dir) or not os.path.isdir(exp_ep_dir):
                if match_mode == "strict":
                    raise FileNotFoundError(f"Missing episode dir. gt={gt_ep_dir} pred={exp_ep_dir}")
                continue

            ok = True
            sec_items = []
            for sec, image_idx in zip(secs, image_idxs):
                gt_sec_img_path = os.path.join(gt_ep_dir, f"{int(image_idx)}.png")
                exp_sec_img_path = os.path.join(exp_ep_dir, f"{int(image_idx)}.png")
                if not os.path.isfile(gt_sec_img_path) or not os.path.isfile(exp_sec_img_path):
                    ok = False
                    break
                sec_items.append((sec, gt_sec_img_path, exp_sec_img_path))

            if not ok:
                if match_mode == "strict":
                    raise FileNotFoundError(f"Missing frame(s) under gt={gt_ep_dir} pred={exp_ep_dir}")
                continue

            for sec, gt_sec_img_path, exp_sec_img_path in sec_items:
                gt_sec_img = transforms.ToTensor()(Image.open(gt_sec_img_path).convert("RGB")).unsqueeze(0)
                exp_sec_img = transforms.ToTensor()(Image.open(exp_sec_img_path).convert("RGB")).unsqueeze(0)

                gt_batch[sec].append(gt_sec_img)
                exp_batch[sec].append(exp_sec_img)

            batch_pairs += 1

        if batch_pairs <= 0:
            continue
        total_pairs += batch_pairs

        for sec in secs:
            if len(gt_batch[sec]) == 0:
                continue

            sec_gt_batch = torch.cat(gt_batch[sec], dim=0).to('cuda', non_blocking=True)
            sec_exp_batch = torch.cat(exp_batch[sec], dim=0).to('cuda', non_blocking=True)

            if sec_exp_batch.shape[-2:] != sec_gt_batch.shape[-2:]:
                sec_exp_batch = F.interpolate(sec_exp_batch, size=sec_gt_batch.shape[-2:], mode="bicubic", align_corners=False,antialias=True)

            n = int(sec_gt_batch.shape[0])
            if n <= 0:
                continue

            lpips_val = float(lpips_fn(sec_gt_batch, sec_exp_batch).detach().cpu().item())
            dreamsim_val = float(dreamsim_fn(sec_gt_batch, sec_exp_batch).detach().cpu().item())
            dino_val = float(dino_fn(sec_gt_batch, sec_exp_batch).detach().cpu().item())
            depth_val = float(depth_fn(sec_gt_batch, sec_exp_batch).detach().cpu().item())

            if eval_type == 'rollout':
                tag = f'frame{sec}'
            else:
                tag = f'{sec}s'

            metric_logger.meters[f'{dataset_name}_{eval_name}_lpips_{tag}'].update(lpips_val, n=n)
            metric_logger.meters[f'{dataset_name}_{eval_name}_dreamsim_{tag}'].update(dreamsim_val, n=n)
            metric_logger.meters[f'{dataset_name}_{eval_name}_dino_{tag}'].update(dino_val, n=n)
            metric_logger.meters[f'{dataset_name}_{eval_name}_depth_l1_{tag}'].update(depth_val, n=n)

            fid_gt = sec_gt_batch.clamp(0.0, 1.0)
            fid_exp = sec_exp_batch.clamp(0.0, 1.0)

            if fid_fn is not None:
                fid_fn[sec].update(images=fid_gt, is_real=True)
                fid_fn[sec].update(images=fid_exp, is_real=False)
            
    if fid_fn is not None:
        fid_fn = _sync_fid_metrics(fid_fn, secs, device="cuda")

        for sec in secs:
            if eval_type == 'rollout':
                tag = f'frame{sec}'
            else:
                tag = f'{sec}s'
            metric_logger.meters[f'{dataset_name}_{eval_name}_fid_{tag}'].update(fid_fn[int(sec)].compute().item(), n=1)

def save_metric_to_disk(metric_logger, log_p):
    metric_logger.synchronize_between_processes()
    if not dist.is_main_process():
        return
    log_stats = {k: float(meter.global_avg) for k, meter in metric_logger.meters.items()}
    with open(log_p, 'w') as json_file:
        json.dump(log_stats, json_file, indent=4)

def main(args):
    dist.init_distributed()
    device = 'cuda'

    # Loading Datasets
    dataset_names = args.datasets.split(',')
    
    secs = np.array([2**i for i in range(0, args.num_sec_eval)])
    
    # Initialize Loss Functions
    print("Initializing Metrics...")
    lpips_fn = get_loss_fn('lpips', secs, device)
    dreamsim_fn = get_loss_fn('dreamsim', secs, device)
    dino_fn = get_loss_fn('dino', secs, device)
    depth_fn = get_loss_fn('depth', secs, device)

    for dataset_name in dataset_names:
        gt_dataset_dir = os.path.join(args.gt_dir, dataset_name)
        exp_dataset_dir = os.path.join(args.exp_dir, dataset_name)
        
        if 'rollout' in args.eval_types:
            for rollout_fps in args.rollout_fps_values:
                try:
                    metric_logger = dist.MetricLogger(delimiter="  ")
                    print("Evaluating rollout", rollout_fps, dataset_name)
                    
                    eval_name = f'rollout_{rollout_fps}fps'
                    gt_dataset_rollout_dir = os.path.join(gt_dataset_dir, eval_name)
                    exp_dataset_rollout_dir = os.path.join(exp_dataset_dir, eval_name)
                    
                    # FID needs separate instance per sec usually if using torchmetrics accum style
                    rollout_fid_fn = get_loss_fn('fid', secs, device)
                    rollout_loss_fns = (lpips_fn, dreamsim_fn, rollout_fid_fn, dino_fn, depth_fn)
                    
                    with torch.no_grad():
                        evaluate(args, dataset_name, 'rollout', metric_logger, rollout_loss_fns, 
                                 gt_dataset_rollout_dir, exp_dataset_rollout_dir, secs, rollout_fps)
                    
                    output_fn = os.path.join(args.exp_dir, f'{dataset_name}_{eval_name}.json')
                    save_metric_to_disk(metric_logger, output_fn)
                except Exception as e:
                    print(f"Error in rollout eval: {e}")
                    import traceback
                    traceback.print_exc()

        if 'time' in args.eval_types:
            try:
                metric_logger = dist.MetricLogger(delimiter="  ")
                print("Evaluating time", dataset_name)
                eval_name = 'time'
                gt_dataset_time_dir = os.path.join(gt_dataset_dir, eval_name)
                exp_dataset_time_dir = os.path.join(exp_dataset_dir, eval_name)
                
                time_fid_fn = get_loss_fn('fid', secs, device)
                time_loss_fns = (lpips_fn, dreamsim_fn, time_fid_fn, dino_fn, depth_fn)
                
                with torch.no_grad():
                    evaluate(args, dataset_name, eval_name, metric_logger, time_loss_fns, 
                             gt_dataset_time_dir, exp_dataset_time_dir, secs, None)
                
                output_fn = os.path.join(args.exp_dir, f'{dataset_name}_{eval_name}.json')
                save_metric_to_disk(metric_logger, output_fn)
            except Exception as e:
                print(f"Error in time eval: {e}")
                import traceback
                traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--batch_size", type=int, default=32, help="batch size (decrease if OOM)")
    parser.add_argument("--eval_types", type=str, default='time,rollout', help="evaluations")
    parser.add_argument("--disable_fid", type=int, default=0, help="disable FID computation")
    parser.add_argument("--gt_dir", type=str, default=None, help="gt directory")
    parser.add_argument("--exp_dir", type=str, default=None, help="experiment directory")
    parser.add_argument("--num_sec_eval", type=int, default=5, help="number of seconds to eval")
    parser.add_argument("--datasets", type=str, default=None, help="dataset name")
    parser.add_argument("--input_fps", type=int, default=4, help="input fps")
    parser.add_argument("--rollout_fps_values", type=str, default='1,4', help="rollout fps values")
    parser.add_argument("--exp", type=str, default=None, help="experiment name")
    parser.add_argument("--match_mode", type=str, default="auto", help="GT/PRED matching strategy: strict|intersection|auto")
    parser.add_argument("--max_trajs", type=int, default=0, help="Max number of trajectories to evaluate (0 means no limit)")
    parser.add_argument("--rollout_last_idx", type=int, default=63, help="In rollout mode, last frame idx to evaluate (e.g., 63), default 63")
    
    args = parser.parse_args()
    
    args.rollout_fps_values = [int(fps) for fps in args.rollout_fps_values.split(',')]
    args.eval_types = args.eval_types.split(',')
    
    main(args)