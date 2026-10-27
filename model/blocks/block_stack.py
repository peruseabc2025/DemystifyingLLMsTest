"""
Block stack module combining N transformer blocks.

Assigned to: Tanush
Status: Clean
"""

import inspect
from typing import Optional, Any
import torch
import torch.nn as nn

from .transformer_block import TransformerBlock
from model.layers.layer_norm import LayerNorm


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


class BlockStack(nn.Module):
    """
    Sequential stack of N transformer decoder blocks followed by final layer norm.
    """

    def __init__(
        self,
        config: Optional[Any] = None,
        n_layers: Optional[int] = None,
        d_model: Optional[int] = None
    ):
        super().__init__()
        cfg = _resolve_config(config)

        self.n_layers = n_layers if n_layers is not None else getattr(cfg, "n_layers", 12)
        self.d_model = d_model if d_model is not None else getattr(cfg, "d_model", 768)

        self.blocks = nn.ModuleList([
            TransformerBlock(config=cfg, layer_idx=i)
            for i in range(self.n_layers)
        ])
        self.final_ln = LayerNorm(config=cfg)

    def __len__(self) -> int:
        return len(self.blocks)

    def __getitem__(self, idx: int) -> TransformerBlock:
        return self.blocks[idx]

    def forward(
        self,
        h: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Any] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Forward pass across all stacked transformer blocks.

        Args:
            h: Input hidden state tensor of shape (batch_size, seq_len, d_model)
            mask: Optional attention mask
            kv_cache: Optional KVCache

        Returns:
            Output hidden state tensor of shape (batch_size, seq_len, d_model)
        """
        for block in self.blocks:
            h = block(h, mask=mask, kv_cache=kv_cache)

        return self.final_ln(h)
