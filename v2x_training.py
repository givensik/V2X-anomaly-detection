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
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

from v2x_preprocessing import V2XDataPreprocessor, save_preprocessor

# -------------------------------
# Model
# -------------------------------
class AutoEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int] = [128, 64, 32]):
        super().__init__()
        enc, prev = [], input_dim
        for h in hidden_dims:
            enc += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(0.2)]
            prev = h
        self.encoder = nn.Sequential(*enc)
        dec, rev = [], list(hidden_dims)[::-1]
        for i, h in enumerate(rev):
            if i == len(rev) - 1:
                dec += [nn.Linear(prev, input_dim), nn.Tanh()]
            else:
                dec += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(0.2)]
            prev = h
        self.decoder = nn.Sequential(*dec)
    def forward(self, x):
        z = self.encoder(x); xhat = self.decoder(z); return xhat

class V2XDataset(Dataset):
    def __init__(self, X, y): self.X = torch.FloatTensor(X); self.y = torch.FloatTensor(y)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

# -------------------------------
# Training utilities
# -------------------------------
def train_autoencoder(train_loader, input_dim, epochs=50, lr=1e-3, device='cpu'):
    model = AutoEncoder(input_dim)
    criterion = nn.MSELoss()
    opt = optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    losses = []
    for ep in range(epochs):
        model.train(); tot = 0.0
        for batch, _ in train_loader:
            batch = batch.to(device)
            b, s, f = batch.shape
            flat = batch.view(b, -1)
            opt.zero_grad()
            rec = model(flat)
            loss = criterion(rec, flat)
            loss.backward(); opt.step()
            tot += loss.item()
        losses.append(tot / max(1, len(train_loader)))
    return model, losses

# Threshold computation을 위한 함수
def compute_threshold(model, val_loader, percentile=95, device='cpu'):
    model.eval()
    errs = []
    with torch.no_grad():
        for batch, _ in val_loader:
            batch = batch.to(device)
            b, s, f = batch.shape
            flat = batch.view(b, -1)
            rec = model(flat)
            err = torch.mean((flat - rec) ** 2, dim=1)
            errs.extend(err.cpu().numpy())
    thr = float(np.percentile(errs, percentile))
    return thr, np.array(errs)

# -------------------------------
# End-to-end train entry
# -------------------------------
def run_training(
    # train parameters
    v2aix_csv_path: str = "out/v2aix_preprocessed.csv",
    veremi_csv_path: str = "out/veremi_preprocessed.csv",
    out_dir: str = "artifacts",
    sequence_length: int = 10,
    batch_size: int = 32,
    epochs: int = 70,
    lr: float = 1e-3,
    percentile: float = 95.0,
    random_state: int = 42,
):
    # 디렉터리 및 장치 설정
    os.makedirs(out_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


    pre = V2XDataPreprocessor()
    print("Loading V2AIX (from CSV) ...")
    v2aix_df = pd.read_csv(v2aix_csv_path)
    print("Loading VeReMi (from CSV) ...")
    veremi_df = pd.read_csv(veremi_csv_path)

    # Preprocess features
    print("Preprocessing features ...")
    df = pd.concat([v2aix_df, veremi_df], ignore_index=True)
    df = pre.preprocess_features(df)


    # 시퀀스화, 
    X, y = pre.create_sequences(df, sequence_length=sequence_length)
    # 각 station_id별 시간순으로 슬라이딩 윈도우(길이 S) → (N, S, F) 생성
    # 시퀀스 라벨 y는 윈도우 내에 1(공격)이 한 번이라도 있으면 1

    print(f"Sequences: {X.shape}, Labels: {y.shape}")
    # y.sum()          # 공격 시퀀스 개수
    # (y==0).sum()     # 정상 시퀀스 개수
    # y.mean()         # 공격 시퀀스 비율
    print(y.sum(), "attacker sequences", (y==0).sum(), "normal sequences", f"{y.mean()*100:.2f}% attacker sequences")
    
    # 데이터 분할 
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=random_state, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=random_state, stratify=y_temp
    )

    # 정상 데이터만 학습
    normal_idx = np.where(y_train == 0)[0]
    X_train_n = X_train[normal_idx]; y_train_n = y_train[normal_idx]
    print(f"Train on normal sequences: {len(X_train_n)}")

    train_loader = DataLoader(V2XDataset(X_train_n, y_train_n), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(V2XDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(V2XDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

    input_dim = X.shape[1] * X.shape[2]
    model, losses = train_autoencoder(train_loader, input_dim, epochs=epochs, lr=lr, device=device)

    thr, val_errs = compute_threshold(model, val_loader, percentile=percentile, device=device)
    print(f"Threshold (p{percentile}): {thr:.6f}")

    # Evaluate on test to report AUC using errors
    model.eval()
    all_errs, all_labels = [], []
    with torch.no_grad():
        for batch, labels in test_loader:
            b, s, f = batch.shape
            flat = batch.view(b, -1).to(device)
            rec = model(flat)
            err = torch.mean((flat - rec) ** 2, dim=1).cpu().numpy()
            all_errs.extend(err); all_labels.extend(labels.numpy())
    auc = float(roc_auc_score(np.array(all_labels), np.array(all_errs))) if len(set(all_labels)) > 1 else float('nan')
    print(f"AUC-ROC (errors): {auc:.4f}")

    # Save artifacts
    torch.save(model.state_dict(), os.path.join(out_dir, "model.pth"))
    save_preprocessor(pre, os.path.join(out_dir, "preprocessor.pkl"))
    with open(os.path.join(out_dir, "training_meta.json"), "w") as f:
        json.dump({
            "sequence_length": sequence_length,
            "input_dim": input_dim,
            "threshold": thr,
            "percentile": percentile,
            "auc_on_test_errors": auc
        }, f, indent=2)

    # Plot training loss
    plt.figure(figsize=(6,4))
    plt.plot(losses)
    plt.title('Training Loss'); plt.xlabel('Epoch'); plt.ylabel('MSE Loss')
    loss_path = os.path.join(out_dir, "training_loss.png")
    plt.tight_layout(); plt.savefig(loss_path, dpi=200, bbox_inches='tight'); plt.close()


    return {
        "artifacts_dir": out_dir,
        "threshold": thr,
        "auc_on_test_errors": auc,
        "loss_plot": loss_path
    }

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Train V2X AutoEncoder anomaly detector")
    ap.add_argument("--v2aix_csv_path", type=str, required=True)
    ap.add_argument("--veremi_csv_path", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="artifacts")
    ap.add_argument("--sequence_length", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--percentile", type=float, default=95.0)
    ap.add_argument("--random_state", type=int, default=42)
    args = ap.parse_args()
    run_training(**vars(args))
