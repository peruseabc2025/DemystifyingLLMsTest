"""Module implementation."""
import torch
import torch.nn.functional as F


def apply_temperature(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    Apply temperature scaling to logits.
    """
    if temperature < 0:
        raise ValueError("temperature must be non-negative")
    if temperature == 0:
        return F.one_hot(logits.argmax(dim=-1), num_classes=logits.size(-1)).to(logits.dtype)
    return F.softmax(logits / temperature, dim=-1)
