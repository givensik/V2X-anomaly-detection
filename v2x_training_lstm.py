import os
import json
import numpy as np
import pandas as pd
from typing import List
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, fbeta_score
import matplotlib.pyplot as plt
from v2x_preprocessing import V2XDataPreprocessor

# -------------------------------
# Model (LSTM-AutoEncoder)
# -------------------------------
class LSTMAutoEncoder(nn.Module):
    def __init__(self, input_dim: int, sequence_length: int, hidden_dim: int = 128, n_layers: int = 2):
        super(LSTMAutoEncoder, self).__init__()
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=0.2
        )
        self.decoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=input_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=0.2
        )

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_dim)
        # Encoder -> Decoder
        _, (hidden, cell) = self.encoder(x)
        # Decoder의 입력으로 Encoder의 마지막 hidden state를 사용
        # (batch_size, sequence_length, hidden_dim) 형태로 변환
        decoder_input = hidden[-1].unsqueeze(1).repeat(1, x.size(1), 1)
        reconstructed, _ = self.decoder(decoder_input)
        return reconstructed

class V2XDataset(Dataset):
    def __init__(self, X, y): self.X = torch.FloatTensor(X); self.y = torch.FloatTensor(y)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

# -------------------------------
# Training utilities
# -------------------------------
def train_lstm_autoencoder(model, train_loader, epochs=50, lr=1e-3, device='cpu'):
    criterion = nn.MSELoss()
    opt = optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    losses = []
    for ep in range(epochs):
        model.train()
        total_loss = 0.0
        for batch, _ in train_loader:
            batch = batch.to(device)
            # LSTM은 데이터를 평탄화하지 않고 (batch, seq_len, features) 그대로 사용
            opt.zero_grad()
            reconstructed = model(batch)
            loss = criterion(reconstructed, batch) # 원본 시퀀스와 비교
            loss.backward()
            opt.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        losses.append(avg_loss)
        print(f'Epoch [{ep+1}/{epochs}], Loss: {avg_loss:.6f}')
    return model, losses

def find_optimal_threshold_fbeta(model, val_loader, y_val, beta=1.5, device='cpu'):
    model.eval()
    errors = []
    with torch.no_grad():
        for batch, _ in val_loader:
            batch = batch.to(device)
            reconstructed = model(batch)
            # 시퀀스 전체에 대한 재구성 오류 계산
            err = torch.mean((batch - reconstructed) ** 2, dim=[1, 2])
            errors.extend(err.cpu().numpy())

    errors = np.array(errors)
    labels = y_val.numpy()
    threshold_candidates = np.linspace(errors.min(), errors.max(), 1000)
    best_fbeta, best_threshold = 0.0, 0.0

    for thr in threshold_candidates:
        preds = (errors > thr).astype(int)
        if len(np.unique(labels)) > 1:
            fbeta = fbeta_score(labels, preds, beta=beta)
            if fbeta > best_fbeta:
                best_fbeta = fbeta
                best_threshold = thr
    print(f"Optimal Threshold (F-beta={beta}): {best_threshold:.6f} with F-beta score: {best_fbeta:.4f}")
    return best_threshold, errors

# -------------------------------
# Main Training Function
# -------------------------------
def run_training(
    v2aix_csv_path: str = "out/v2aix_preprocessed.csv",
    veremi_csv_path: str = "out/veremi_preprocessed.csv",
    out_dir: str = "artifacts_lstm",
    sequence_length: int = 10,
    batch_size: int = 64,
    epochs: int = 50,
    lr: float = 1e-3,
    random_state: int = 42,
):
    os.makedirs(out_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    feature_columns = [
        'pos_x', 'pos_y', 'pos_z', 'spd_x', 'spd_y', 'spd_z',
        'heading', 'speed', 'acceleration', 'curvature'
    ]
    print(f"Using {len(feature_columns)} features.")

    print("Loading preprocessed data from CSV...")
    v2aix_df = pd.read_csv(v2aix_csv_path)
    veremi_df = pd.read_csv(veremi_csv_path)

    pre = V2XDataPreprocessor(feature_columns=feature_columns)
    
    # 시퀀스 생성
    print("Creating sequences...")
    X_v2aix, y_v2aix = pre.create_sequences(v2aix_df, sequence_length=sequence_length)
    X_veremi, y_veremi = pre.create_sequences(veremi_df, sequence_length=sequence_length)

    # # 시퀀스 결합
    # X = np.concatenate([X_v2aix, X_veremi], axis=0)
    # y = np.concatenate([y_v2aix, y_veremi], axis=0)
    # print(f"Combined sequences: {X.shape}, Labels: {y.shape}")

    # # 데이터 분할(v2aix/veremi 섞음)
    # X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=random_state, stratify=y)
    # X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=random_state, stratify=y_temp)
    # # 이건 뭐냐 그 normal만 뽑는거
    # normal_idx = np.where(y_train == 0)[0]
    # X_train_n, y_train_n = X_train[normal_idx], y_train[normal_idx]


    # V2AIX로 정상학습, VeReMi로 검증/테스트
    # V2AIX로 정상학습
    X_train_n = X_v2aix
    y_train_n = y_v2aix
    print(f"Train on PURE normal sequences (V2AIX only): {len(X_train_n)}")

    # VeReMi로 검증/테스트셋 분리
    X_val, X_test, y_val, y_test = train_test_split(
        X_veremi, y_veremi, test_size=0.5, random_state=random_state, stratify=y_veremi
    )
    print(f"Validation sequences (from VeReMi): {X_val.shape}")
    print(f"Test sequences (from VeReMi): {X_test.shape}")
    

    train_loader = DataLoader(V2XDataset(X_train_n, y_train_n), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(V2XDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(V2XDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

    num_features = X_v2aix.shape[2]
    model = LSTMAutoEncoder(input_dim=num_features, sequence_length=sequence_length)
    
    print("Training LSTM-AutoEncoder...")
    model, losses = train_lstm_autoencoder(model, train_loader, epochs=epochs, lr=lr, device=device)
    
    print("Finding optimal threshold...")
    thr, _ = find_optimal_threshold_fbeta(model, val_loader, torch.FloatTensor(y_val), beta=1.5, device=device)

    print("Evaluating on test set...")
    test_errors, test_labels = [], []
    with torch.no_grad():
        for batch, labels in test_loader:
            batch = batch.to(device)
            reconstructed = model(batch)
            err = torch.mean((batch - reconstructed) ** 2, dim=[1, 2])
            test_errors.extend(err.cpu().numpy())
            test_labels.extend(labels.numpy())
    
    auc = roc_auc_score(test_labels, test_errors) if len(np.unique(test_labels)) > 1 else float('nan')
    print(f"AUC-ROC on Test Set: {auc:.4f}")

    torch.save(model.state_dict(), os.path.join(out_dir, "model_lstm.pth"))
    with open(os.path.join(out_dir, "training_meta_lstm.json"), "w") as f:
        json.dump({
            "sequence_length": sequence_length,
            "input_dim": num_features,
            "threshold": float(thr),
            "auc_on_test_errors": auc
        }, f, indent=2)

    plt.figure(figsize=(6,4))
    plt.plot(losses)
    plt.title('Training Loss'); plt.xlabel('Epoch'); plt.ylabel('MSE Loss')
    plt.savefig(os.path.join(out_dir, "training_loss_lstm.png"), dpi=200)
    plt.close()

if __name__ == "__main__":
    run_training()