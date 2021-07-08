import torch.nn as nn
import torch
from .multi_head_attention import MultiHeadAttention
from .positionwide_feedforward import PositionwiseFeedforward


class AttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, dropout, device):
        super().__init__()
        self.layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttention(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforward(hid_dim, pf_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask):
        # src = [batch size, src len, hid dim]
        # src_mask = [batch size, src len]
        # self attention
        _src, attn = self.self_attention(src, src, src, src_mask)
        # dropout, residual connection and layer norm
        src = self.layer_norm(src + self.dropout(_src))
        # src = [batch size, src len, hid dim]
        # positionwise feedforward
        _src = self.positionwise_feedforward(src)
        # dropout, residual and layer norm
        src = self.layer_norm(src + self.dropout(_src))
        # src = [batch size, src len, hid dim]
        return src, attn


class Attention(nn.Module):
    def __init__(self, hid_dim, n_layers, n_heads, pf_dim, dropout, device):
        super().__init__()
        self.device = device
        self.layers = nn.ModuleList([AttentionLayer(hid_dim, n_heads, pf_dim, dropout, device)
                                     for _ in range(n_layers)])

    def forward(self, src, src_mask):
        # src = [batch size, src len, hid dim]
        # src_mask = [batch size, src len]
        for layer in self.layers:
            src, attn = layer(src, src_mask)
        # src = [batch size, src len, hid dim]
        return src, attn.squeeze()