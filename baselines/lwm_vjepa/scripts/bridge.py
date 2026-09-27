"""Lossless T=1 ABI bridge to the unchanged image-only evaluation planner."""
from PIL import Image

class ImageTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, rgb):
        value = self.transform(Image.fromarray(rgb))
        if tuple(value.shape) != (3, 1, 384, 384):
            raise ValueError('expected native single-frame VJEPA preprocessing')
        return value.squeeze(1)

class ModelABI:
    def __init__(self, model):
        self.model = model

    @property
    def max_len(self):
        return self.model.max_len

    def roll_out(self, now, goal, *args, **kwargs):
        return self.model.roll_out(now.unsqueeze(2), goal.unsqueeze(2), *args, **kwargs)

    def get_reward(self, now, goal, *args, **kwargs):
        return self.model.get_reward(now.unsqueeze(2), goal.unsqueeze(2), *args, **kwargs)
