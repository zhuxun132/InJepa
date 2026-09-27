"""Byte validation for the pinned released source and pretrained checkpoints."""
from pathlib import Path
import hashlib,json


def validate_assets(manifest_path):
    manifest_path=Path(manifest_path).resolve()
    rows=json.loads(manifest_path.read_text())['assets']
    if not rows:raise ValueError('empty asset manifest')
    result={}
    for row in rows:
        name=row['path'];p=Path(name)
        if not p.is_absolute():p=manifest_path.parent/p
        p=p.resolve()
        if name in result:raise ValueError('duplicate asset manifest entry')
        if not p.is_file():raise FileNotFoundError(p)
        if p.stat().st_size!=row['bytes']:raise ValueError(f'asset size mismatch: {p}')
        h=hashlib.sha256()
        with p.open('rb') as f:
            for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
        if h.hexdigest()!=row['sha256']:raise ValueError(f'asset SHA mismatch: {p}')
        result[name]=p
    return result


class RGBBackend:
    def __init__(self,planner,transform,*,device='cpu'):
        self.planner,self.transform,self.device=planner,transform,device

    def reset(self):
        self.planner.reset()

    def plan(self,current_rgb,goal_rgb):
        from .adapter import _rgb
        now=self.transform(_rgb(current_rgb)).unsqueeze(0).to(self.device)
        goal=self.transform(_rgb(goal_rgb)).unsqueeze(0).to(self.device)
        return self.planner.plan(now,goal)
