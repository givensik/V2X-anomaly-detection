# train_autoencoder.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
from z_models.autoencoder import V2XAutoEncoder
from z_models.lstm_autoencoder import LSTMAutoEncoder

# 설정
BATCH_SIZE = 128
EPOCHS = 50
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 데이터 로드
X_train = np.load("z_data/train_X.npy")  # shape: (N, 20, 6)
y_train = np.load("z_data/train_y.npy")

# Tensor 변환
train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32))
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# 모델 초기화
# model = V2XAutoEncoder(input_dim=6, seq_len=20).to(DEVICE)
model = LSTMAutoEncoder(input_dim=6).to(DEVICE)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 학습 루프
print("🚀 Training AutoEncoder...")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch in train_loader:
        x = batch[0].to(DEVICE)
        output = model(x)
        loss = criterion(output, x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1:02d}/{EPOCHS} | Loss: {avg_loss:.6f}")

# 모델 저장
os.makedirs("z_models", exist_ok=True)
torch.save(model.state_dict(), "z_models/autoencoder.pth")
print("✅ Model saved to z_models/autoencoder.pth")
