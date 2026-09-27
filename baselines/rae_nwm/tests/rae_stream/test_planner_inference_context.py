"""The adapter must preserve the official evaluate invocation context."""
from contextlib import contextmanager
import sys
from types import SimpleNamespace
import numpy as np
import torch
from rae_stream.habitat_policy import OfficialRAEPlannerBackend


def test_official_generate_actions_runs_without_grad_inside_cuda_bfloat16_autocast(monkeypatch):
    module=sys.modules[__name__]; seen={}; active=[]
    @contextmanager
    def autocast(device_type, *, enabled=True, dtype=None):
        seen['amp']=(device_type,enabled,dtype);active.append(True)
        try:yield
        finally:active.pop()
    monkeypatch.setattr(torch.amp,'autocast',autocast)
    monkeypatch.setattr(module,'save_planning_pred',lambda *a,**k:None,raising=False)
    class Evaluator:
        args=SimpleNamespace(save_preds=False)
        def generate_actions(self,*args):
            seen['grad']=torch.is_grad_enabled();seen['active']=bool(active)
            module.save_planning_pred(deltas=torch.zeros((1,8,3)))
    backend=OfficialRAEPlannerBackend(Evaluator(),transform=lambda im:torch.zeros((3,2,2)),torch_module=torch)
    assert backend.plan(np.zeros((4,2,2,3),dtype=np.uint8),np.zeros((2,2,3),dtype=np.uint8))==(0.,0.,0.)
    assert seen['grad'] is False
    assert seen['active'] is True
    assert seen['amp']==('cuda',True,torch.bfloat16)
    assert torch.is_grad_enabled()
