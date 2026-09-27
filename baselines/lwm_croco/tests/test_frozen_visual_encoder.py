import sys
from pathlib import Path
import torch
from torch import nn
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))
from lwm_stream.freezing import freeze_visual_encoder

def test_freeze_preserves_output_rng_and_fusion_learning():
    torch.manual_seed(4)
    model=nn.Module(); c=model.croco=nn.Module()
    c.patch_embed=nn.Linear(3,4); c.enc_blocks=nn.Sequential(nn.Linear(4,4)); c.enc_blocks.requires_grad_(False)
    c.enc_norm=nn.LayerNorm(4); c.enc_pos_embed=nn.Parameter(torch.zeros(4), requires_grad=False)
    c.decoder_embed=nn.Linear(4,4); c.prediction_head=nn.Linear(4,1)
    x=torch.randn(5,3)
    def forward():
        return c.prediction_head(c.decoder_embed(c.enc_norm(c.enc_blocks(c.patch_embed(x)))))
    before=forward().detach().clone(); rng=torch.get_rng_state().clone()
    original={n:p.detach().clone() for n,p in model.named_parameters()}
    receipt=freeze_visual_encoder(model)
    assert receipt['newly_frozen_parameters']==24
    assert torch.equal(rng,torch.get_rng_state())
    assert torch.equal(before,forward().detach())
    opt=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=.01)
    forward().square().mean().backward(); opt.step()
    for name,p in model.named_parameters():
        if name.startswith(('croco.patch_embed.','croco.enc_')):
            assert p.grad is None and torch.equal(p,original[name])
        else:
            assert p.grad is not None and torch.isfinite(p.grad).all()
            assert not torch.equal(p,original[name])
