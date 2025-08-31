# z_models/autoencoder.py

import torch
import torch.nn as nn

class V2XAutoEncoder(nn.Module):
    def __init__(self, input_dim: int = 6, seq_len: int = 20, hidden_dim: int = 64):
        super(V2XAutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(seq_len * input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, seq_len * input_dim),
            nn.Unflatten(1, (seq_len, input_dim))
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out
