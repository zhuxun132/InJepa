"""Image preprocessing and trajectory-loading helpers for inference."""

import json

import numpy as np
import torch
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_image_transform():
    """Resize/normalize transform matching the training pipeline."""
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def load_kmeans_trajectories(path, normalize=True):
    """Load the k-means trajectory codebook used as world-model candidates.

    Args:
        path: JSON file holding an array of shape ``(K, 63, 3)``.
        normalize: if True, scale trajectories by 0.1 to match the model inputs.

    Returns:
        tensor of shape ``(K, 63, 3)``.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    actions = torch.tensor(np.array(data), dtype=torch.float32)
    if normalize:
        actions = actions * 0.1
    return actions
