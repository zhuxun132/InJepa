"""End-to-end inference demo for LWM (latent world model).

Given a current observation image and a goal observation image, this script:
  1. samples candidate trajectories with the policy (``ARPlusPolicy``);
  2. scores them with the world model (``LatentWorldModel``);
  3. returns the trajectory that the world model deems closest to the goal.

It also shows how to call the world model directly on the k-means trajectory
codebook (``eval_wm``), which is the pure world-model inference mode.

Usage:
    python inference.py --now-img path/to/now.jpg --goal-img path/to/goal.jpg
"""

import argparse
import os

import torch
from PIL import Image

from lwm import ActionTokenizer, ARPlusPolicy, LatentWorldModel
from lwm.preprocess import get_image_transform, load_kmeans_trajectories

DEFAULT_ROOT = os.path.dirname(os.path.abspath(__file__))

DEFAULT_PATHS = {
    "wm_ckpt": os.path.join(DEFAULT_ROOT, "weights", "world_model.pt"),
    "policy_ckpt": os.path.join(DEFAULT_ROOT, "weights", "ar_plus.pt"),
    "tokenizer_center": os.path.join(DEFAULT_ROOT, "tokenizer", "zed2_64.json"),
    "kmeans_traj": os.path.join(DEFAULT_ROOT, "tokenizer", "zed2_traj.json"),
}


def build_world_model(ckpt_path, device):
    model = LatentWorldModel()
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval()
    return model


def build_policy(ckpt_path, device):
    model = ARPlusPolicy()
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="LWM inference demo")
    parser.add_argument("--now-img", required=True, help="current observation image")
    parser.add_argument("--goal-img", required=True, help="goal observation image")
    parser.add_argument("--wm-ckpt", default=DEFAULT_PATHS["wm_ckpt"])
    parser.add_argument("--policy-ckpt", default=DEFAULT_PATHS["policy_ckpt"])
    parser.add_argument("--tokenizer-center", default=DEFAULT_PATHS["tokenizer_center"])
    parser.add_argument("--kmeans-traj", default=DEFAULT_PATHS["kmeans_traj"])
    parser.add_argument("--num-sample", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)

    transform = get_image_transform()

    img_now = transform(Image.open(args.now_img).convert("RGB")).unsqueeze(0).to(device)
    img_obj = transform(Image.open(args.goal_img).convert("RGB")).unsqueeze(0).to(device)

    tokenizer = ActionTokenizer(args.tokenizer_center)
    kmeans_trajs = load_kmeans_trajectories(args.kmeans_traj, normalize=True).to(device)

    # ---- world model in "pure" mode: pick from the k-means codebook ----
    wm = build_world_model(args.wm_ckpt, device)
    kmeans_idx, point_idx = wm.eval_wm(img_now, img_obj, kmeans_trajs)
    wm_best = kmeans_trajs[kmeans_idx[0], point_idx[0]]
    print(f"[world model] best codebook end pose (x, y, yaw, normalized): {wm_best.tolist()}")

    # ---- policy + world model: sample candidates, score, pick best ----
    policy = build_policy(args.policy_ckpt, device)
    token_seqs = policy.roll_out(img_now, img_obj, num_sample=args.num_sample, temperature=1.0)
    token_seqs = token_seqs.view(1 * args.num_sample, policy.max_len)

    actions, lengths = tokenizer.batch_decode(token_seqs)              # (S, 63, 3), (S,)
    actions = actions.view(1, args.num_sample, 63, 3)
    actions_normal = 0.1 * actions

    reward = wm.get_reward(img_now, img_obj, actions_normal, lengths)  # (1, S)
    best_sample = reward[0].argmax().item()
    best_traj = actions_normal[0, best_sample, : lengths[best_sample]]

    print(f"[policy + world model] best sample {best_sample}/{args.num_sample}, "
          f"length {lengths[best_sample].item()}")
    print(f"[policy + world model] best end pose (x, y, yaw, normalized): "
          f"{best_traj[-1].tolist()}")


if __name__ == "__main__":
    main()
