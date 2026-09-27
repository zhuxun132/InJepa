"""Optional CPU-only recording of actual Habitat RGB observations."""
from __future__ import annotations
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import uuid
import numpy as np


class FirstPersonVideoRecorder:
    def __init__(self, path, *, fps=4, episode_key=None, identities=None):
        if isinstance(fps, bool) or not math.isfinite(float(fps)) or float(fps) <= 0:
            raise ValueError('video fps must be positive and finite')
        self.path = Path(path)
        self.metadata_path = self.path.with_suffix('.json')
        if self.path.exists() or self.metadata_path.exists():
            raise FileExistsError(str(self.path))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.temp = self.path.with_name('.' + self.path.stem + '.' + uuid.uuid4().hex + '.mp4')
        self.fps = float(fps)
        self.episode_key = copy.deepcopy(episode_key)
        self.identities = copy.deepcopy(identities)
        self.frames = []
        self.writer = None
        self.shape = None
        self.goal_sha256 = None
        self.closed = False

    def __call__(self, event):
        if self.closed:
            raise RuntimeError('video recorder is closed')
        from PIL import Image, ImageDraw
        rgb = np.array(event['rgb'], copy=True)
        goal = np.array(event['imagegoal'], copy=True)
        for value in (rgb, goal):
            if value.dtype != np.uint8 or value.ndim != 3 or value.shape[2] != 3:
                raise ValueError('video requires real uint8 HWC RGB arrays')
        goal_sha = hashlib.sha256(goal.tobytes()).hexdigest()
        if self.goal_sha256 is not None and goal_sha != self.goal_sha256:
            raise ValueError('goal RGB changed during recorded episode')
        self.goal_sha256 = goal_sha
        h, w = rgb.shape[:2]
        size = (w + 160 + (w % 2), max(h, 216) + (max(h, 216) % 2))
        if self.shape is not None and size != self.shape:
            raise ValueError('video frame dimensions changed')
        canvas = Image.new('RGB', size, (20, 20, 20))
        canvas.paste(Image.fromarray(rgb), (0, 0))
        thumb = Image.fromarray(goal)
        thumb.thumbnail((152, 116))
        canvas.paste(thumb, (w + 4, 24))
        draw = ImageDraw.Draw(canvas)
        draw.text((w + 4, 4), 'GOAL IMAGE', fill='white')
        draw.text((w + 4, 142), f"Step: {event['step']}", fill='white')
        draw.text((w + 4, 158), f"Action: {event['action'] or 'RESET'}", fill='white')
        draw.text((w + 4, 190), f'Playback: {self.fps:g} fps', fill='white')
        if self.writer is None:
            import imageio_ffmpeg
            self.writer = imageio_ffmpeg.write_frames(str(self.temp), size, fps=self.fps, codec='libx264', pix_fmt_in='rgb24', pix_fmt_out='yuv420p', macro_block_size=1, output_params=['-preset', 'veryfast', '-threads', '1'])
            self.writer.send(None)
            self.shape = size
        self.writer.send(np.asarray(canvas).copy())
        self.frames.append({'step': event['step'], 'action': event['action'], 'phase': event['phase'], 'rgb_sha256': hashlib.sha256(rgb.tobytes()).hexdigest()})

    def close(self):
        if self.closed:
            raise RuntimeError('video recorder already closed')
        self.closed = True
        if self.writer is None:
            raise ValueError('cannot publish empty video')
        self.writer.close()
        if not self.temp.is_file() or self.temp.stat().st_size == 0:
            raise RuntimeError('encoder did not produce a video')
        metadata = {'recorded': True, 'frame_count': len(self.frames), 'fps': self.fps, 'timing': 'playback_not_wall_clock', 'episode_key': self.episode_key, 'identities': self.identities, 'goal_sha256': self.goal_sha256, 'frames': self.frames, 'path': str(self.path), 'sha256': hashlib.sha256(self.temp.read_bytes()).hexdigest()}
        # Hard-link publication cannot replace another run's evidence.
        os.link(self.temp, self.path)
        self.temp.unlink()
        from j2j_iclr_experiments.common.artifacts import create_once_json
        create_once_json(self.metadata_path, metadata)
        return metadata
