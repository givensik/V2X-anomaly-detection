# z_models/lstm_autoencoder.py

import torch
import torch.nn as nn

class LSTMAutoEncoder(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=32, latent_dim=16, num_layers=1):
        super(LSTMAutoEncoder, self).__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.latent = nn.Linear(hidden_dim, latent_dim)
        self.decoder_input = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.LSTM(hidden_dim, input_dim, num_layers, batch_first=True)

    def forward(self, x):
        enc_out, _ = self.encoder(x)                          # [B, T, H]
        latent = self.latent(enc_out[:, -1, :])               # [B, L]
        repeated = self.decoder_input(latent).unsqueeze(1).repeat(1, x.size(1), 1)  # [B, T, H]
        dec_out, _ = self.decoder(repeated)                   # [B, T, F]
        return dec_out
