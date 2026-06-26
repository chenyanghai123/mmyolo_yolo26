"""Minimal checkpoint loader (loads state_dict with strict=False)."""
import torch


def load_checkpoint(model, checkpoint_path, strict=False, map_location="cpu"):
    """Load a checkpoint into model (state_dict only, no runner state)."""
    ckpt = torch.load(checkpoint_path, map_location=map_location)
    if "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    elif "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt
    # Strip 'module.' prefix if present (DDP)
    new_state = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[len("module."):]
        new_state[k] = v
    missing, unexpected = model.load_state_dict(new_state, strict=strict)
    print(f"[load_checkpoint] missing={len(missing)} unexpected={len(unexpected)} (strict={strict})")
    return {"missing_keys": missing, "unexpected_keys": unexpected}
