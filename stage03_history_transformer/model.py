"""Target-history-only, deterministic K=1 Transformer; no map or neighbors yet."""
import torch
from torch import nn


class HistoryTransformer(nn.Module):
    def __init__(self, d_model=128, heads=4, layers=3, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(6, d_model)
        self.time_embed = nn.Parameter(torch.randn(1, 50, d_model) * 0.02)  
        block = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=heads, dim_feedforward=4*d_model,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            block, num_layers=layers, enable_nested_tensor=False,
        )
        # Encoder clones would otherwise start with identical layer weights.
        for layer in self.encoder.layers:
            for parameter in layer.parameters():
                if parameter.ndim > 1:
                    nn.init.xavier_uniform_(parameter)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU(),
            nn.Linear(d_model, 60*2),
        )

    def forward(self, hist, hist_valid):
        tokens = self.input_proj(hist) + self.time_embed
        # Dataset: True means valid. PyTorch padding mask: True means ignore.
        memory = self.encoder(tokens, src_key_padding_mask=~hist_valid)
        # All selected focal histories include valid timestep 49.
        return self.head(memory[:, 49]).reshape(-1, 60, 2)


def position_loss(pred, future, future_valid):
    squared_distance = (pred - future).square().sum(-1)
    return (squared_distance * future_valid).sum() / future_valid.sum().clamp_min(1)
