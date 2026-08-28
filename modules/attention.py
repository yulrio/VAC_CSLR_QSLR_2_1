import torch
import torch.nn as nn


class TemporalSelfAttention(nn.Module):
    def __init__(self, d_model=1024, num_heads=4, dropout=0.1):
        super(TemporalSelfAttention, self).__init__()
        assert d_model % num_heads == 0, f"d_model {d_model} must be divisible by num_heads {num_heads}"
        
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=False
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, lengths):
        T, B, _ = x.shape
        mask = torch.arange(T, device=x.device).unsqueeze(0) >= lengths.to(x.device).unsqueeze(1)
        attn_out, _ = self.attn(x, x, x, key_padding_mask=mask)
        out = self.norm(x + self.dropout(attn_out))
        return out
