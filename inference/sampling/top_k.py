"""Top-k sampling. Assigned: Sarthak. Bug: M10"""
import torch


def apply_top_k(logits: torch.Tensor, k: int) -> torch.Tensor:
    """
    Mask logits outside top-k.
    """
    if k <= 0 or k >= logits.size(-1):
        return logits

    _, indices = torch.topk(logits, k, largest=True, dim=-1)
    keep = torch.zeros_like(logits, dtype=torch.bool).scatter(-1, indices, True)
    return logits.masked_fill(~keep, float("-inf"))
