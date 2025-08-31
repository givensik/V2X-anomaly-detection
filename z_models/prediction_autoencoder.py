# z_models/prediction_autoencoder.py
import torch
import math
import torch.nn as nn

# class PredictionAutoEncoder(nn.Module):
#     def __init__(self, input_dim=6, hidden_dim=64, num_layers=1, seq_len=20, pred_len=3):
#         super(PredictionAutoEncoder, self).__init__()
#         self.seq_len = seq_len
#         self.pred_len = pred_len
#         self.input_dim = input_dim

#         # LSTM 인코더: 입력 시퀀스 처리
#         self.encoder = nn.LSTM(
#             input_size=input_dim,
#             hidden_size=hidden_dim,
#             num_layers=num_layers,
#             batch_first=True
#         )

#         # 선형 디코더: hidden state → pred_len * input_dim 차원으로 변환
#         self.decoder = nn.Linear(hidden_dim, pred_len * input_dim)

#     def forward(self, x):
#         # x: (batch_size, seq_len, input_dim)
#         _, (hidden, _) = self.encoder(x)  # hidden: (num_layers, batch_size, hidden_dim)

#         # 마지막 layer의 hidden state 사용
#         last_hidden = hidden[-1]  # (batch_size, hidden_dim)

#         # 디코딩 후 reshape
#         decoded = self.decoder(last_hidden)  # (batch_size, pred_len * input_dim)
#         output = decoded.view(-1, self.pred_len, self.input_dim)  # (batch_size, pred_len, input_dim)

#         return output

# dropout 적용 버젼
# class PredictionAutoEncoder(nn.Module):
#     def __init__(self, input_dim=6, hidden_dim=128, num_layers=1, seq_len=20, pred_len=3, dropout=0.3):
#         super(PredictionAutoEncoder, self).__init__()
#         self.seq_len = seq_len
#         self.pred_len = pred_len
#         self.input_dim = input_dim

#         # LSTM Encoder with Dropout
#         self.encoder = nn.LSTM(
#             input_size=input_dim,
#             hidden_size=hidden_dim,
#             num_layers=num_layers,
#             dropout=dropout if num_layers > 1 else 0.0,  # Dropout은 num_layers > 1일 때만 적용됨
#             batch_first=True
#         )

#         # Dropout before decoder
#         self.dropout = nn.Dropout(dropout)

#         # Linear Decoder
#         self.decoder = nn.Linear(hidden_dim, pred_len * input_dim)

#     def forward(self, x):
#         # x: (batch_size, seq_len, input_dim)
#         _, (hidden, _) = self.encoder(x)  # hidden: (num_layers, batch_size, hidden_dim)
#         last_hidden = hidden[-1]          # (batch_size, hidden_dim)

#         dropped = self.dropout(last_hidden)
#         decoded = self.decoder(dropped)   # (batch_size, pred_len * input_dim)
#         output = decoded.view(-1, self.pred_len, self.input_dim)  # (batch_size, pred_len, input_dim)

#         return output
    

# 계층 증가
# class PredictionAutoEncoder(nn.Module):
#     def __init__(self, input_dim=6, hidden_dim=128, seq_len=20, pred_len=1, num_layers=2, dropout=0.3):
#         super(PredictionAutoEncoder, self).__init__()
#         self.seq_len = seq_len
#         self.pred_len = pred_len

#         self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)

#         self.decoder = nn.LSTM(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)

#         self.output_layer = nn.Linear(hidden_dim, input_dim)

#     def forward(self, x):
#         batch_size = x.size(0)
#         _, (h_n, c_n) = self.encoder(x)

#         # Decoder input: repeat last hidden state pred_len times
#         decoder_input = torch.zeros(batch_size, self.pred_len, h_n.size(2)).to(x.device)
#         decoder_output, _ = self.decoder(decoder_input, (h_n, c_n))

#         # Apply output layer to each timestep
#         out = self.output_layer(decoder_output)
#         return out

# positional encoding 추가
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # even index
        pe[:, 1::2] = torch.cos(position * div_term)  # odd index
        self.pe = pe.unsqueeze(0)  # (1, max_len, d_model)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)].to(x.device)  # (batch, seq_len, d_model)
        return x


class PredictionAutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, seq_len, pred_len, num_layers=2, dropout=0.2):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.input_dim = input_dim  # 생성자에 추가
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim, max_len=seq_len)
        self.encoder = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, pred_len * input_dim)
        )

    def forward(self, x):  # x: (batch, seq_len, input_dim)
        x = self.input_proj(x)             # → (batch, seq_len, hidden_dim)
        x = self.pos_encoder(x)            # → + positional info
        _, (h_n, _) = self.encoder(x)      # → h_n: (num_layers, batch, hidden_dim)
        last_hidden = h_n[-1]              # → (batch, hidden_dim)
        out = self.decoder(last_hidden)    # → (batch, pred_len * input_dim)
        return out.view(-1, self.pred_len, self.input_dim)  # → (batch, pred_len, input_dim)