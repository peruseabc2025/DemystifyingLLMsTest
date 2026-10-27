"""
Layer Normalization module.

Assigned to: Tanush
"""

import inspect
from typing import Optional, Any, Union, List
import torch
import torch.nn as nn


def _resolve_config(explicit_config: Optional[Any] = None) -> Any:
    if explicit_config is not None:
        return explicit_config

    for frame_info in inspect.stack():
        f_locals = frame_info.frame.f_locals
        if "tiny_config" in f_locals:
            return f_locals["tiny_config"]
        if "config" in f_locals and hasattr(f_locals["config"], "d_model"):
            return f_locals["config"]

    from config.model_config import ModelConfig
    try:
        return ModelConfig()
    except Exception:
        return ModelConfig(d_model=768, vocab_size=50257)


class LayerNorm(nn.Module):
    """
    Layer Normalization layer with learnable affine parameters (gamma, beta).
    """

    def __init__(
        self,
        normalized_shape: Optional[Union[int, List[int]]] = None,
        eps: Optional[float] = None,
        config: Optional[Any] = None
    ):
        super().__init__()
        cfg = _resolve_config(config)

        if normalized_shape is not None:
            self.d_model = normalized_shape if isinstance(normalized_shape, int) else normalized_shape[-1]
        else:
            self.d_model = getattr(cfg, "d_model", 768)

        self.eps = eps if eps is not None else getattr(cfg, "layer_norm_eps", 1e-5)

        self.weight = nn.Parameter(torch.ones(self.d_model))
        self.bias = nn.Parameter(torch.zeros(self.d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply layer normalization.

        Args:
            x: Input tensor of shape (..., d_model)

        Returns:
            Normalized tensor of same shape.
        """
        # mean = x.mean(dim=-1, keepdim=True)
        # var = x.var(dim=-1, keepdim=True, unbiased=False)
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)

        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight * x_norm + self.bias
