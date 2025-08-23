import os
import json
import numpy as np
import pandas as pd
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, fbeta_score, accuracy_score
import matplotlib.pyplot as plt

from v2x_preprocessing_lstm import V2XDataPreprocessor, fuse_scores  # ★ rule late-fusion 유틸 재사용

# -------------------------------
# Model (LSTM-AutoEncoder)
# -------------------------------
class LSTMAutoEncoder(nn.Module):
    def __init__(self, input_dim, sequence_length, hidden_dim=128, n_layers=2, dropout=0.1):
        super().__init__()
        # encoder: bidirectional
        self.enc = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim,
                           num_layers=n_layers, batch_first=True,
                           dropout=dropout, bidirectional=True)
        # bridge: 2*hidden -> hidden
        self.bridge = nn.Linear(2*hidden_dim, hidden_dim)
        # decoder: hidden -> hidden (repeat T)
        self.dec = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim,
                           num_layers=n_layers, batch_first=True, dropout=dropout)
        # head: hidden -> input_dim
        self.head = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        # x: (B,T,F)
        _, (h, c) = self.enc(x)     # h: (2*n_layers, B, hidden)
        h_last_fwd = h[-2,:,:]      # 마지막 layer forward
        h_last_bwd = h[-1,:,:]      # 마지막 layer backward
        z = torch.cat([h_last_fwd, h_last_bwd], dim=-1)  # (B, 2H)
        z = torch.tanh(self.bridge(z))                   # (B, H)
        z_seq = z.unsqueeze(1).repeat(1, x.size(1), 1)   # (B,T,H)
        y, _ = self.dec(z_seq)                           # (B,T,H)
        recon = self.head(y)                             # (B,T,F)
        return recon
# -------------------------------
# Forecaster (LSTM)
# -------------------------------
class LSTMForecaster(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, n_layers=2, dropout=0.3):
        super().__init__()
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
            bidirectional=False
        )
        self.norm_h = nn.LayerNorm(hidden_dim)   # 안정화
        self.head   = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        # x: (B, T, D). 예측 대상은 x[:, 1:, :]
        x_in = x[:, :-1, :]                      # (B, T-1, D)
        enc_out, _ = self.encoder(x_in)          # (B, T-1, H)
        z = self.norm_h(enc_out)
        y_hat = self.head(z)                     # (B, T-1, D)
        return y_hat
    
    
    

class V2XDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

# -------------------------------
# Training utilities
# -------------------------------
def train_lstm_autoencoder(model, train_loader, val_loader=None, epochs=50, lr=1e-3, device='cpu', out_dir='artifacts_lstm'):
    """
    개선된 AutoEncoder 학습 함수
    - Validation 추가
    - Early stopping 추가
    - Learning rate scheduler 추가
    - Gradient clipping 추가
    """
    criterion = nn.SmoothL1Loss(reduction='none')  # Huber loss
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)  # Weight decay 추가
    
    # Learning rate scheduler 추가
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5)
    
    model.to(device)
    train_losses = []
    val_losses = []
    
    best_val_loss = float('inf')
    patience = 10
    bad_epochs = 0
    
    for ep in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for batch, _ in train_loader:
            batch = batch.to(device)
            opt.zero_grad()
            recon = model(batch)
            loss = criterion(recon, batch).mean()  # 평균 계산
            loss.backward()
            
            # Gradient clipping 추가
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            opt.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation (validation loader가 있는 경우)
        val_loss = float('inf')
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch, _ in val_loader:
                    batch = batch.to(device)
                    recon = model(batch)
                    loss = criterion(recon, batch).mean()
                    val_loss += loss.item()
            val_loss /= len(val_loader)
            val_losses.append(val_loss)
            
            # Learning rate scheduler step
            scheduler.step(val_loss)
            
            # Early stopping 체크
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                bad_epochs = 0
                # Best model 저장
                os.makedirs(out_dir, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(out_dir, "best_autoencoder.pth"))
            else:
                bad_epochs += 1
            
            print(f'Epoch [{ep+1}/{epochs}]  train_loss={train_loss:.6f}  val_loss={val_loss:.6f}  lr={opt.param_groups[0]["lr"]:.2e}')
            
            if bad_epochs >= patience:
                print(f"Early stopping at epoch {ep+1}")
                break
        else:
            print(f'Epoch [{ep+1}/{epochs}]  train_loss={train_loss:.6f}')
    
    # Best model 로드
    if val_loader is not None and os.path.exists(os.path.join(out_dir, "best_autoencoder.pth")):
        model.load_state_dict(torch.load(os.path.join(out_dir, "best_autoencoder.pth")))
        print("Loaded best model from early stopping")
    
    return model, train_losses, val_losses

# -------------------------------
# Forecaster training (not used in testing, but here for completeness)
# -------------------------------
def train_forecaster(
    model, train_loader, val_loader,
    epochs=50, lr=1e-3, weight_decay=1e-4,
    device='cpu', out_dir='artifacts_lstm'
):
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5)
    best_val = float('inf'); patience = 10; bad = 0
    losses = []

    for ep in range(epochs):
        # ---- train ----
        model.train(); tot = 0.0
        for xb, _ in train_loader:
            xb = xb.to(device)
            opt.zero_grad()
            yb = model(xb)                # (B, T-1, D)
            loss = loss_fn(yb, xb)        # vs xb[:,1:,:]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item()
        tr_loss = tot / max(1, len(train_loader))
        losses.append(tr_loss)

        # ---- val (간단 proxy) ----
        model.eval(); vtot = 0.0
        with torch.no_grad():
            for vb, _ in val_loader:
                vb = vb.to(device)
                vtot += loss_fn(model(vb), vb).item()
        val_loss = vtot / max(1, len(val_loader))
        sched.step(val_loss)

        if val_loss < best_val:
            best_val, bad = val_loss, 0
            os.makedirs(out_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(out_dir, "best_forecaster.pth"))
        else:
            bad += 1

        print(f"Ep {ep+1:03d}/{epochs}  train={tr_loss:.5f}  val={val_loss:.5f}  lr={opt.param_groups[0]['lr']:.2e}")
        if bad >= patience:
            print("Early stop."); break

    return model, losses




# ★ combined-score로 임계값 탐색 (AE오차 + rule)
def find_optimal_threshold_fbeta_combined(ae_errors: np.ndarray,
                                          rule_seq_scores: np.ndarray,
                                          y_true: np.ndarray,
                                          beta: float = 1.5,
                                          alpha: float = 0.7) -> Tuple[float, np.ndarray]:
    """
    ae_errors: (N,) 시퀀스별 AE 재구성오차
    rule_seq_scores: (N,) 시퀀스별 rule 점수 (0~1, 전처리에서 계산)
    y_true: (N,) 라벨
    """
    combined = fuse_scores(ae_errors, rule_seq_scores, alpha=alpha)  # 0~1로 정규화된 결합 점수
    thr_candidates = np.linspace(combined.min(), combined.max(), 1000)
    best_f, best_thr = 0.0, float(combined.mean())  # 초기값

    for thr in thr_candidates:
        preds = (combined > thr).astype(int)
        if len(np.unique(y_true)) > 1:
            f = fbeta_score(y_true, preds, beta=beta)
            if f > best_f:
                best_f, best_thr = f, thr

    print(f"[Fusion] Best threshold={best_thr:.6f} (F{beta}={best_f:.4f})")
    return best_thr, combined

# add: 자동 튜닝 함수
def tune_alpha_and_threshold(ae_errors_val, rule_seq_val, y_val, betas=(1.5,), alphas=np.linspace(0,1,21)):
    best = {"alpha":0.7, "beta":1.5, "thr":0.0, "fb":-1.0, "auc":-1.0}
    for alpha in alphas:
        comb = fuse_scores(ae_errors_val, rule_seq_val, alpha=alpha)
        # threshold는 comb 분포 위에서 탐색
        thr_cands = np.linspace(comb.min(), comb.max(), 400)
        for beta in betas:
            from sklearn.metrics import fbeta_score, roc_auc_score
            auc = roc_auc_score(y_val, comb) if len(np.unique(y_val))>1 else np.nan
            # F-beta 최대 임계값
            fb_best, thr_best = -1.0, comb.mean()
            for thr in thr_cands:
                preds = (comb > thr).astype(int)
                fb = fbeta_score(y_val, preds, beta=beta)
                if fb > fb_best:
                    fb_best, thr_best = fb, thr
            if fb_best > best["fb"]:
                best = {"alpha":float(alpha), "beta":float(beta), "thr":float(thr_best), "fb":float(fb_best), "auc":float(auc)}
    print(f"[TUNE] alpha={best['alpha']:.2f}, beta={best['beta']}, thr={best['thr']:.6f}, F{best['beta']}={best['fb']:.4f}, AUC={best['auc']:.4f}")
    return best


# -------------------------------
# Loss function with motion weighting (for forecaster)
# Huber 손실 + 모션 가중치(선택)
# -------------------------------
criterion = nn.SmoothL1Loss(reduction='none')  # Huber
    
def loss_fn(y_hat, x):
   # 타깃: x[:, 1:, :]
    target = x[:, 1:, :]
    raw = criterion(y_hat, target).mean(dim=2)  # (B, T-1)

    # 모션 가중치 (speed>0.5 m/s에 가중 ↑). 입력의 speed는 8D 중 하나였지?
    # feature order: [...,'speed',...] 라면 그 index를 speed_idx로 지정 (0부터 시작)
    speed_idx = 2  # 예시: 네 피처 순서에 맞춰 바꿔줘!
    spd = x[:, 1:, speed_idx].abs()
    w = (spd > 0.5).float() * 0.7 + 0.3         # [0.3, 1.0]
    return (raw * w).mean()

# -------------------------------
# Main Training Function
# -------------------------------
def run_training(
    v2aix_csv_path: str = "out/v2aix_preprocessed.csv",
    veremi_csv_path: str = "out/veremi_preprocessed.csv",
    out_dir: str = "artifacts_lstm",
    sequence_length: int = 20,        # ★ 전처리 예제와 맞춤
    batch_size: int = 64,
    epochs: int = 50,
    lr: float = 1e-3,
    alpha: float = 0.7,               # ★ late-fusion 가중치 (AE 비중)
    beta: float = 1.5,                # ★ 임계값 선택 F-beta
    random_state: int = 42,
):
    os.makedirs(out_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("Loading preprocessed CSVs...")
    v2aix_df = pd.read_csv(v2aix_csv_path)
    veremi_df = pd.read_csv(veremi_csv_path)

    # ★ 학습 피처는 전처리에서 사용한 Δ/파생/상대좌표 그대로 사용
    pre = V2XDataPreprocessor()  # feature_columns 기본값 사용 (dpos_x, dspeed, ..., rel_pos_y)
    used_features = [c for c in pre.feature_columns if c in v2aix_df.columns]
    print(f"Using features ({len(used_features)}): {used_features}")

    # ----------------- 시퀀스 생성 -----------------
    print("Creating sequences...")
    # V2AIX: 정상만으로 훈련
    X_v2aix, y_v2aix = pre.create_sequences(v2aix_df, sequence_length=sequence_length)
    print(f"V2AIX sequences: {X_v2aix.shape}, labels unique={np.unique(y_v2aix)}")

    # VeReMi: 검증/테스트 + rule_seq 함께
    X_veremi, y_veremi, rule_veremi = pre.create_sequences_with_rules(veremi_df, sequence_length=sequence_length)
    print(f"VeReMi sequences: {X_veremi.shape}, rule_seq shape={rule_veremi.shape}")

    # ----------------- 분할 -----------------
    # 훈련: V2AIX만 (정상으로 가정)
    X_train_n = X_v2aix
    y_train_n = y_v2aix
    print(f"Train on PURE normal sequences (V2AIX only): {len(X_train_n)}")

    # VeReMi → 검증/테스트 분할 (order 유지용 shuffle=False는 아래 DataLoader에서)
    X_val, X_test, y_val, y_test, r_val, r_test = train_test_split(
        X_veremi, y_veremi, rule_veremi,
        test_size=0.5, random_state=random_state, stratify=y_veremi
    )
    print(f"Validation sequences (VeReMi): {X_val.shape}, Test sequences: {X_test.shape}")

    # ----------------- DataLoader -----------------
    train_loader = DataLoader(V2XDataset(X_train_n, y_train_n), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(V2XDataset(X_val, y_val), batch_size=batch_size, shuffle=False)  # ★순서보존
    test_loader  = DataLoader(V2XDataset(X_test, y_test), batch_size=batch_size, shuffle=False) # ★순서보존

    # ----------------- 모델 -----------------
    num_features = X_train_n.shape[2]
    
    # LSTM-AutoEncoder (개선된 학습 함수 사용)
    model = LSTMAutoEncoder(input_dim=num_features, sequence_length=sequence_length)
    print("Training LSTM-AutoEncoder with validation...")
    model, train_losses, val_losses = train_lstm_autoencoder(
        model, train_loader, val_loader, epochs=epochs, lr=lr, device=device, out_dir=out_dir
    )
    
    # 학습 곡선 저장
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    if val_losses:
        plt.plot(val_losses, label='Val Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(out_dir, "training_loss.png"))
    plt.close()

    # ----------------- 검증: AE오차 계산 & late-fusion 임계값 -----------------
    print("Validating & selecting threshold with late-fusion...")
    model.eval()
    val_errors = []
    with torch.no_grad():
        for batch, _ in val_loader:
            batch = batch.to(device)
            recon = model(batch)  # AutoEncoder reconstruction
            err = torch.mean((batch - recon) ** 2, dim=[1, 2])  # MSE reconstruction error
            val_errors.extend(err.cpu().numpy())
    val_errors = np.array(val_errors)

    # 검증 AUC 출력 (AutoEncoder용)
    ae_norm = (val_errors - np.percentile(val_errors, 1)) / (np.percentile(val_errors, 99) - np.percentile(val_errors, 1) + 1e-12)
    auc_ae   = roc_auc_score(y_val, ae_norm) if len(np.unique(y_val))>1 else float('nan')
    auc_rule = roc_auc_score(y_val, r_val)    if len(np.unique(y_val))>1 else float('nan')
    print(f"[Diag] Val AUC - AE:{auc_ae:.3f}  Rule:{auc_rule:.3f}")

    best = tune_alpha_and_threshold(val_errors, r_val, y_val, betas=(1.0,1.5,2.0), alphas=np.linspace(0,1,21))
    alpha = best["alpha"]
    thr   = best["thr"]


    # ----------------- 테스트 -----------------
    print("Evaluating on test set (late-fusion)...")
    test_errors, test_labels = [], []
    with torch.no_grad():
        for batch, labels in test_loader:
            batch = batch.to(device)
            recon = model(batch)  # AutoEncoder reconstruction
            err = torch.mean((batch - recon) ** 2, dim=[1, 2])  # MSE reconstruction error
            test_errors.extend(err.cpu().numpy())
            test_labels.extend(labels.numpy())
    test_errors = np.array(test_errors)
    test_labels = np.array(test_labels)

    # late-fusion
    combined_test = fuse_scores(test_errors, r_test, alpha=alpha)
    auc = roc_auc_score(test_labels, combined_test) if len(np.unique(test_labels)) > 1 else float('nan')

    # 임계값 적용 Accuracy/F-beta
    preds_test = (combined_test > thr).astype(int)
    acc = accuracy_score(test_labels, preds_test)
    fbeta = fbeta_score(test_labels, preds_test, beta=beta) if len(np.unique(test_labels)) > 1 else float('nan')

    print(f"AUC-ROC (fusion) on Test: {auc:.4f}")
    print(f"Accuracy@thr (fusion) on Test: {acc:.4f}")
    print(f"F{beta}@thr (fusion) on Test: {fbeta:.4f}")

    # ----------------- 저장 -----------------
    os.makedirs(out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(out_dir, "model_lstm.pth"))
    with open(os.path.join(out_dir, "training_meta_lstm.json"), "w") as f:
        json.dump({
            "sequence_length": sequence_length,
            "input_dim": int(num_features),
            "alpha_fusion": float(alpha),
            "fbeta_beta": float(beta),
            "threshold_fusion": float(thr),
            "auc_test_fusion": float(auc),
            "acc_test_fusion": float(acc),
            "fbeta_test_fusion": float(fbeta),
            "used_features": used_features
        }, f, indent=2)

    plt.figure(figsize=(6,4))
    plt.plot(train_losses, label='Train Loss')
    if val_losses:
        plt.plot(val_losses, label='Val Loss')
    plt.title('Training Loss'); plt.xlabel('Epoch'); plt.ylabel('MSE Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "training_loss_lstm.png"), dpi=200)
    plt.close()

if __name__ == "__main__":
    run_training()
