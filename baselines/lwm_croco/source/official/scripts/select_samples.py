"""Pick visual world-model samples from the zed2 test set.

For each test sample we run the world model in `eval_wm` mode and measure two
quantities:

* ``dist_xy``  — endpoint error (meters): predicted vs. ground-truth end pose.
* ``y_range``  — lateral (left/right) spread of the ground-truth trajectory in
  meters, used as a proxy for "large left/right viewpoint change".

We keep samples with *small* error and *large* lateral change, sort by error,
and save the start/goal images plus a ``samples.json`` metadata file. The saved
images are later loaded by ``demo/test_world_model.py``.

Usage (run from the training repo checkout so the data paths resolve):
    python scripts/select_samples.py \
        --pkl /path/to/data/zed2/test_eval.pkl \
        --data-root /path/to/data/zed2/data \
        --out demo/samples
"""

import argparse
import json
import os
import pickle
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from lwm import LatentWorldModel
from lwm.preprocess import get_image_transform, load_kmeans_trajectories


def parse_args():
    p = argparse.ArgumentParser(description="select zed2 world-model samples")
    p.add_argument("--pkl", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--ckpt", default="weights/world_model.pt")
    p.add_argument("--traj", default="tokenizer/zed2_traj.json")
    p.add_argument("--out", default="demo/samples")
    p.add_argument("--max-dist", type=float, default=0.5, help="max endpoint error (m)")
    p.add_argument("--min-yrange", type=float, default=2.0, help="min lateral spread (m)")
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    os.makedirs(args.out, exist_ok=True)

    transform = get_image_transform()
    codebook = load_kmeans_trajectories(args.traj, normalize=True).to(device)

    wm = LatentWorldModel().to(device).eval()
    wm.load_state_dict(torch.load(args.ckpt, map_location="cpu", weights_only=True))

    with open(args.pkl, "rb") as f:
        data = pickle.load(f)

    def full_path(path):
        traj_name, img_name = path.split("/")
        return os.path.join(args.data_root, traj_name, "img", img_name)

    def load_img(path):
        return transform(Image.open(full_path(path)).convert("RGB"))

    rows = []
    for idx in tqdm(range(len(data)), desc="scoring"):
        item = data[idx]
        img_paths = item["img_path"]
        idx_goal = random.Random(idx).randint(0, 62)

        motions = torch.tensor(np.array(item["waypoints"][1:]), dtype=torch.float32)
        yaw = torch.tensor(np.array(item["yaw"][1:]), dtype=torch.float32).view(63, 1)
        xy_yaw = 0.1 * torch.cat([motions, yaw], dim=-1)  # (63, 3), normalized
        gt_end = xy_yaw[idx_goal].to(device)

        # lateral (left/right) spread of the ground-truth trajectory, in meters
        y_range = float(motions[: idx_goal + 1, 1].max() - motions[: idx_goal + 1, 1].min())

        now = load_img(img_paths[0]).unsqueeze(0).to(device)
        obj = load_img(img_paths[idx_goal + 1]).unsqueeze(0).to(device)

        with torch.no_grad():
            kmeans_idx, point_idx = wm.eval_wm(now, obj, codebook)
        pred_end = codebook[kmeans_idx[0], point_idx[0]]  # (3,), normalized
        dist_xy = float(10 * torch.norm(pred_end[:2] - gt_end[:2]))

        if dist_xy <= args.max_dist and y_range >= args.min_yrange:
            rows.append(
                {
                    "sample_idx": idx,
                    "idx_goal": idx_goal,
                    "dist_xy": round(dist_xy, 4),
                    "y_range": round(y_range, 4),
                    "gt_end_xy": [round(float(10 * gt_end[0]), 4), round(float(10 * gt_end[1]), 4)],
                    "pred_end_xy": [round(float(10 * pred_end[0]), 4), round(float(10 * pred_end[1]), 4)],
                    "start_img_path": img_paths[0],
                    "goal_img_path": img_paths[idx_goal + 1],
                }
            )

    # keep the lowest-error samples
    rows.sort(key=lambda r: r["dist_xy"])
    rows = rows[: args.topk]

    manifest = []
    for i, r in enumerate(rows):
        start_name = f"{i:02d}_start.jpg"
        goal_name = f"{i:02d}_goal.jpg"
        Image.open(full_path(r["start_img_path"])).convert("RGB").save(os.path.join(args.out, start_name), quality=95)
        Image.open(full_path(r["goal_img_path"])).convert("RGB").save(os.path.join(args.out, goal_name), quality=95)

        entry = {
            "sample_idx": r["sample_idx"],
            "idx_goal": r["idx_goal"],
            "dist_xy": r["dist_xy"],
            "y_range": r["y_range"],
            "gt_end_xy": r["gt_end_xy"],
            "pred_end_xy": r["pred_end_xy"],
            "start_img": start_name,
            "goal_img": goal_name,
        }
        manifest.append(entry)

    with open(os.path.join(args.out, "samples.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"kept {len(manifest)} samples -> {args.out}")
    for e in manifest:
        print(f"  [{e['start_img']} -> {e['goal_img']}] idx_goal={e['idx_goal']} "
              f"dist_xy={e['dist_xy']}m y_range={e['y_range']}m")


if __name__ == "__main__":
    main()
