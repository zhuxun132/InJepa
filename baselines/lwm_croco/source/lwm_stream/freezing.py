"""Optional full single-image encoder freeze; leave CroCo fusion unchanged."""
def freeze_visual_encoder(model):
    c = model.croco
    newly = 0
    names = []
    for name, parameter in c.named_parameters():
        if name.startswith(('patch_embed.', 'enc_blocks.', 'enc_norm.')) or name == 'enc_pos_embed':
            newly += parameter.numel() if parameter.requires_grad else 0
            parameter.requires_grad_(False)
            names.append(name)
    if not all(any(n.startswith(prefix) for n in names) for prefix in ('patch_embed.', 'enc_blocks.', 'enc_norm.')):
        raise ValueError('CroCo single-image encoder ABI mismatch')
    return {'newly_frozen_parameters': newly, 'frozen_names': names}
