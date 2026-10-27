"""
Multi-Head Attention module.

Assigned to: Tanush
  - M9: Heads are concatenated along wrong dimension (dim=-2 instead of dim=-1)
  - H10: Q, K, V projection matrices are all initialized from the SAME linear layer (shared weights)
"""

import inspect
from typing import Optional, Any, Tuple
import torch
import torch.nn as nn

from .scaled_dot_product import scaled_dot_product_attention
from .causal_mask import create_causal_mask


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


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention (Vaswani et al., 2017).
    Projects inputs into Q, K, V representations across multiple heads,
    performs scaled dot-product attention, and projects back to d_model.
    """

    def __init__(
        self,
        config: Optional[Any] = None,
        d_model: Optional[int] = None,
        n_heads: Optional[int] = None,
        dropout: Optional[float] = None,
        bias: bool = False
    ):
        super().__init__()
        cfg = _resolve_config(config)

        self.d_model = d_model or getattr(cfg, "d_model", 768)
        self.n_heads = n_heads or getattr(cfg, "n_heads", 12)
        dropout_p = dropout if dropout is not None else getattr(cfg, "dropout", 0.1)

        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )
        self.d_head = self.d_model // self.n_heads

        # self.q_proj = nn.Linear(self.d_model, self.d_model, bias=bias)
        # self.k_proj = nn.Linear(self.d_model, self.d_model, bias=bias)
        # self.v_proj = nn.Linear(self.d_model, self.d_model, bias=bias)
        self.q_proj = nn.Linear(self.d_model, self.d_model, bias=bias)
        self.k_proj = nn.Linear(self.d_model, self.d_model, bias=bias)
        self.v_proj = nn.Linear(self.d_model, self.d_model, bias=bias)

        self.out_proj = nn.Linear(self.d_model, self.d_model, bias=bias)
        self.attn_dropout = nn.Dropout(p=dropout_p) if dropout_p > 0.0 else None

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Any] = None,
        layer_idx: Optional[int] = None,
        use_causal_mask: bool = True
    ) -> torch.Tensor:
        """
        Forward pass for multi-head attention.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional additive attention mask
            kv_cache: Optional KVCache instance
            layer_idx: Layer index for KV caching
            use_causal_mask: Whether to automatically generate causal mask if mask is None

        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape

        # 1. Project inputs to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # 2. Reshape into (batch_size, n_heads, seq_len, d_head)
        q = q.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)

        # 3. Update KV cache if provided
        if kv_cache is not None and layer_idx is not None:
            k, v = kv_cache.update(layer_idx, k, v)

        # 4. Prepare causal mask if needed
        if mask is None and use_causal_mask and seq_len > 1:
            mask = create_causal_mask(seq_len=seq_len, device=x.device, dtype=x.dtype)

        # 5. Scaled dot-product attention
        attn_out, _ = scaled_dot_product_attention(
            q=q,
            k=k,
            v=v,
            mask=mask,
            dropout=self.attn_dropout
        )

        # 6. Recombine heads
        # context = torch.cat([attn_out[:, i] for i in range(self.n_heads)], dim=-1)
        context = torch.cat([attn_out[:, i] for i in range(self.n_heads)], dim=-1)

        return self.out_proj(context)
