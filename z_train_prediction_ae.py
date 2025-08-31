# train_prediction_ae.py

import os
import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from z_models.prediction_autoencoder import PredictionAutoEncoder

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 하이퍼파라미터
INPUT_DIM = 8 # 입력 차원
HIDDEN_DIM = 128
SEQ_LEN = 5
PRED_LEN = 1
NUM_LAYERS = 2
DROPOUT = 0.3
EPOCHS = 50
BATCH_SIZE = 128
LR = 0.001

# 데이터 로드
X_train = np.load("z_data/train_X.npy")  # (N, 20, 8)
Y_train = np.load("z_data/train_Y.npy")  # (N, 3, 8)
print(X_train.shape, Y_train.shape)
X_tensor = torch.tensor(X_train, dtype=torch.float32)
Y_tensor = torch.tensor(Y_train, dtype=torch.float32)

train_dataset = TensorDataset(X_tensor, Y_tensor)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# 모델 초기화
model = PredictionAutoEncoder(
    input_dim=INPUT_DIM,
    hidden_dim=HIDDEN_DIM,
    seq_len=SEQ_LEN,
    pred_len=PRED_LEN,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT
).to(DEVICE)

optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()

# 학습 루프
os.makedirs("z_models", exist_ok=True)
print("Training starts...\n")
for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0
    for X_batch, Y_batch in train_loader:
        X_batch, Y_batch = X_batch.to(DEVICE), Y_batch.to(DEVICE)

        optimizer.zero_grad()
        Y_pred = model(X_batch)
        loss = criterion(Y_pred, Y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X_batch.size(0)

    avg_loss = total_loss / len(train_loader.dataset)
    print(f"[Epoch {epoch:02d}] Loss: {avg_loss:.6f}")

# 저장
torch.save(model.state_dict(), "z_models/prediction_autoencoder.pth")
print("\nModel saved to z_models/prediction_autoencoder.pth")
