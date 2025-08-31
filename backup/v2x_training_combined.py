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
from sklearn.metrics import roc_auc_score, average_precision_score, fbeta_score, accuracy_score
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
def train_lstm_autoencoder(model, train_loader, val_loader=None, epochs=50, lr=1e-3, device='cpu', out_dir='artifacts_lstm_combined'):
    """
    개선된 AutoEncoder 학습 함수
    - Validation 추가
    - Early stopping 추가
    - Learning rate scheduler 추가
    - Gradient clipping 추가
    """
    criterion = nn.MSELoss(reduction='none')  # MSE loss (테스트와 일치)
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)  # Weight decay 추가
    
    # Learning rate scheduler 추가
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=10)
    
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
                    loss = criterion(recon, batch).mean()  # 배치별 평균
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
    device='cpu', out_dir='artifacts_lstm_combined'
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
    
    # 라벨 분포 확인
    unique_labels = np.unique(y_val)
    print(f"[TUNE] Validation labels: {unique_labels}, counts: {[np.sum(y_val == label) for label in unique_labels]}")
    
    if len(unique_labels) <= 1:
        print("[TUNE] Warning: Only one class in validation data, using default values")
        return best
    
    for alpha in alphas:
        comb = fuse_scores(ae_errors_val, rule_seq_val, alpha=alpha)
        # threshold는 comb 분포 위에서 탐색 (고해상도)
        thr_cands = np.linspace(comb.min(), comb.max(), 5000)
        for beta in betas:
            from sklearn.metrics import fbeta_score, roc_auc_score, average_precision_score
            try:
                auc_roc = roc_auc_score(y_val, comb)
                auc_pr = average_precision_score(y_val, comb)
            except:
                auc_roc = np.nan
                auc_pr = np.nan
            
            # F-beta 최대 임계값
            fb_best, thr_best = -1.0, comb.mean()
            for thr in thr_cands:
                preds = (comb > thr).astype(int)
                try:
                    fb = fbeta_score(y_val, preds, beta=beta)
                except:
                    fb = 0.0
                if fb > fb_best:
                    fb_best, thr_best = fb, thr
            
            if fb_best > best["fb"]:
                best = {"alpha":float(alpha), "beta":float(beta), "thr":float(thr_best), "fb":float(fb_best), "auc_roc":float(auc_roc), "auc_pr":float(auc_pr)}
    
    print(f"[TUNE] alpha={best['alpha']:.2f}, beta={best['beta']}, thr={best['thr']:.6f}, F{best['beta']}={best['fb']:.4f}, ROC-AUC={best['auc_roc']:.4f}, PR-AUC={best['auc_pr']:.4f}")
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
    out_dir: str = "artifacts_lstm_combined",
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
    
    # V2AIX: 정상 데이터만 추출
    v2aix_normal = v2aix_df[v2aix_df['is_attacker'] == 0]  # 정상 데이터만
    X_v2aix, y_v2aix = pre.create_sequences(v2aix_normal, sequence_length=sequence_length)
    print(f"V2AIX normal sequences: {X_v2aix.shape}, labels unique={np.unique(y_v2aix)}")

    # VeReMi: 정상 데이터와 공격 데이터 분리
    veremi_normal = veremi_df[veremi_df['is_attacker'] == 0]  # 정상 데이터만
    veremi_attack = veremi_df[veremi_df['is_attacker'] != 0]  # 공격 데이터
    
    # VeReMi 정상 데이터 시퀀스 생성
    X_veremi_norm, y_veremi_norm = pre.create_sequences(veremi_normal, sequence_length=sequence_length)
    print(f"VeReMi normal sequences: {X_veremi_norm.shape}, labels unique={np.unique(y_veremi_norm)}")
    
    # VeReMi 공격 데이터 시퀀스 생성 (rule 포함)
    X_veremi_attack, y_veremi_attack, rule_veremi_attack = pre.create_sequences_with_rules(veremi_attack, sequence_length=sequence_length)
    print(f"VeReMi attack sequences: {X_veremi_attack.shape}, rule_seq shape={rule_veremi_attack.shape}")
    
    # ★ 정상 데이터 합치기 (V2AIX + VeReMi 정상)
    X_normal_combined = np.vstack([X_v2aix, X_veremi_norm])
    y_normal_combined = np.hstack([y_v2aix, y_veremi_norm])
    print(f"Combined normal sequences: {X_normal_combined.shape} (V2AIX: {X_v2aix.shape[0]}, VeReMi: {X_veremi_norm.shape[0]})")
    
    # ★ 데이터 스케일링 적용 (정상 데이터 + 공격 데이터 합쳐서 fit, 모든 데이터에 transform)
    print("Applying unified data scaling...")
    # 정상 데이터와 공격 데이터를 합쳐서 scaler fit (전체 분포 고려)
    X_normal_flat = X_normal_combined.reshape(-1, X_normal_combined.shape[-1])
    X_attack_flat = X_veremi_attack.reshape(-1, X_veremi_attack.shape[-1])
    X_combined = np.vstack([X_normal_flat, X_attack_flat])
    
    print(f"Combined data shape for scaling: {X_combined.shape}")
    pre.scaler.fit(X_combined)
    pre.is_fitted = True
    
    # 모든 데이터에 transform 적용
    X_normal_combined = pre.scaler.transform(X_normal_flat).reshape(X_normal_combined.shape)
    X_veremi_attack = pre.scaler.transform(X_attack_flat).reshape(X_veremi_attack.shape)
    
    print(f"Scaling applied - Normal combined range: [{X_normal_combined.min():.3f}, {X_normal_combined.max():.3f}]")
    print(f"Scaling applied - VeReMi attack range: [{X_veremi_attack.min():.3f}, {X_veremi_attack.max():.3f}]")

    # ----------------- 분할 (개선된 방법) -----------------
    # 전체 데이터를 합쳐서 분할 (정상 + 공격)
    print("Creating unified train/val/test split...")
    
    # 정상 데이터에 라벨 0, 공격 데이터에 라벨 1로 통일
    y_normal_combined = np.zeros(len(X_normal_combined))  # 정상 = 0
    y_veremi_attack = np.ones(len(X_veremi_attack))      # 공격 = 1
    
    # 전체 데이터 합치기
    X_all = np.vstack([X_normal_combined, X_veremi_attack])
    y_all = np.hstack([y_normal_combined, y_veremi_attack])
    r_all = np.hstack([np.zeros(len(y_normal_combined)), rule_veremi_attack])
    
    print(f"Total data - Normal: {len(y_normal_combined)}, Attack: {len(y_veremi_attack)}")
    
    # 전체 데이터를 60/20/20으로 분할 (stratified)
    X_train, X_temp, y_train, y_temp, r_temp = train_test_split(
        X_all, y_all, r_all, test_size=0.4, random_state=random_state, stratify=y_all
    )
    X_val, X_test, y_val, y_test, r_test = train_test_split(
        X_temp, y_temp, r_temp, test_size=0.5, random_state=random_state, stratify=y_temp
    )
    
    print(f"Split results:")
    print(f"  Train: {X_train.shape}, labels: {np.unique(y_train, return_counts=True)}")
    print(f"  Val: {X_val.shape}, labels: {np.unique(y_val, return_counts=True)}")
    print(f"  Test: {X_test.shape}, labels: {np.unique(y_test, return_counts=True)}")
    
    # 훈련 데이터는 정상 데이터만 사용 (AutoEncoder 학습용)
    train_normal_mask = (y_train == 0)
    X_train_norm = X_train[train_normal_mask]
    y_train_norm = y_train[train_normal_mask]
    
    print(f"Training data (normal only): {X_train_norm.shape}")

    # ----------------- DataLoader -----------------
    train_loader = DataLoader(V2XDataset(X_train_norm, y_train_norm), batch_size=batch_size, shuffle=True)
    
    # 검증 데이터 분리 (정상/공격)
    val_normal_mask = (y_val == 0)
    val_attack_mask = (y_val == 1)
    
    X_val_norm = X_val[val_normal_mask]
    y_val_norm = y_val[val_normal_mask]
    X_val_attack = X_val[val_attack_mask]
    y_val_attack = y_val[val_attack_mask]
    r_val_attack = r_temp[val_attack_mask]  # Use r_temp instead of undefined r_val
    
    val_loader_norm = DataLoader(V2XDataset(X_val_norm, y_val_norm), batch_size=batch_size, shuffle=False)
    val_loader_attack = DataLoader(V2XDataset(X_val_attack, y_val_attack), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(V2XDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

    # ----------------- 모델 -----------------
    num_features = X_train_norm.shape[2]
    
    # LSTM-AutoEncoder (개선된 학습 함수 사용)
    model = LSTMAutoEncoder(input_dim=num_features, sequence_length=sequence_length)

    print("Training LSTM-AutoEncoder with combined normal data...")
    model, train_losses, val_losses = train_lstm_autoencoder(
        model, train_loader, val_loader_norm, epochs=epochs, lr=lr, device=device, out_dir=out_dir
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
    val_labels = []
    val_rule_scores = []
    
    # 피처별 가중치 (움직임/변화에 더 민감하게)
    # feature order: ['dpos_x', 'dpos_y', 'dspeed', 'dheading_rad', 'acceleration', 'curvature', 'rel_pos_x', 'rel_pos_y']
    feature_weights = torch.tensor([1.2, 1.2, 1.5, 1.3, 1.2, 1.0, 1.0, 1.0], device=device)  # 움직임 피처에 가중
    
    # 정상 검증 데이터 처리
    with torch.no_grad():
        for batch, labels in val_loader_norm:  # 정상 검증셋
            batch = batch.to(device)
            recon = model(batch)  # AutoEncoder reconstruction
            # 피처별 가중 MSE 계산
            weighted_err = ((batch - recon) ** 2) * feature_weights.view(1, 1, -1)
            err = torch.mean(weighted_err, dim=[1, 2])  # 가중 평균
            val_errors.extend(err.cpu().numpy())
            val_labels.extend(labels.numpy())
            # 정상 데이터는 rule score = 0
            val_rule_scores.extend([0.0] * len(labels))
    
    # 공격 검증 데이터 처리
    with torch.no_grad():
        for batch, labels in val_loader_attack:  # 공격 검증셋
            batch = batch.to(device)
            recon = model(batch)  # AutoEncoder reconstruction
            # 피처별 가중 MSE 계산
            weighted_err = ((batch - recon) ** 2) * feature_weights.view(1, 1, -1)
            err = torch.mean(weighted_err, dim=[1, 2])  # 가중 평균
            val_errors.extend(err.cpu().numpy())
            val_labels.extend(labels.numpy())
            val_rule_scores.extend(r_val_attack[:len(labels)])  # 해당 배치의 rule scores
    
    val_errors = np.array(val_errors)
    val_labels = np.array(val_labels)
    val_rule_scores = np.array(val_rule_scores)
    
    print(f"Validation data - Normal: {np.sum(val_labels == 0)}, Attack: {np.sum(val_labels != 0)}")
    print(f"Validation errors range: [{val_errors.min():.6f}, {val_errors.max():.6f}]")
    print(f"Validation rule scores range: [{val_rule_scores.min():.6f}, {val_rule_scores.max():.6f}]")

    # 진단 AUC (혼합 검증셋 기준)
    ae_norm = (val_errors - np.percentile(val_errors, 1)) / (np.percentile(val_errors, 99) - np.percentile(val_errors, 1) + 1e-12)
    auc_ae_roc   = roc_auc_score(val_labels, ae_norm) if len(np.unique(val_labels))>1 else float('nan')
    auc_ae_pr    = average_precision_score(val_labels, ae_norm) if len(np.unique(val_labels))>1 else float('nan')
    auc_rule_roc = roc_auc_score(val_labels, val_rule_scores)    if len(np.unique(val_labels))>1 else float('nan')
    auc_rule_pr  = average_precision_score(val_labels, val_rule_scores)    if len(np.unique(val_labels))>1 else float('nan')
    print(f"[Diag] Val AUC (Mixed) - AE: ROC={auc_ae_roc:.3f}, PR={auc_ae_pr:.3f}")
    print(f"[Diag] Val AUC (Mixed) - Rule: ROC={auc_rule_roc:.3f}, PR={auc_rule_pr:.3f}")

    # 튜닝 후 임계값 적용 (고해상도 α 탐색)
    # 전체 범위 + 0.90-1.00 구간 세밀 탐색
    alphas_coarse = np.linspace(0, 0.9, 19)  # 0.0~0.9, 0.05 간격
    alphas_fine = np.linspace(0.90, 1.00, 11)  # 0.90~1.00, 0.01 간격
    alphas_combined = np.concatenate([alphas_coarse, alphas_fine])
    
    best = tune_alpha_and_threshold(val_errors, val_rule_scores, val_labels, betas=(1.0,1.5,2.0), alphas=alphas_combined) # 혼합 검증셋
    alpha = best["alpha"]
    thr   = best["thr"]

    # ----------------- 테스트 -----------------
    print("Evaluating on test set (late-fusion)...")
    test_errors, test_labels = [], []
    with torch.no_grad():
        for batch, labels in test_loader:
            batch = batch.to(device)
            recon = model(batch)  # AutoEncoder reconstruction
            # 피처별 가중 MSE 계산 (검증과 동일)
            weighted_err = ((batch - recon) ** 2) * feature_weights.view(1, 1, -1)
            err = torch.mean(weighted_err, dim=[1, 2])  # 가중 평균
            test_errors.extend(err.cpu().numpy())
            test_labels.extend(labels.numpy())
    test_errors = np.array(test_errors)
    test_labels = np.array(test_labels)

    # late-fusion
    combined_test = fuse_scores(test_errors, r_test, alpha=alpha)
    
    # 테스트 데이터 라벨 분포 확인
    unique_test_labels = np.unique(test_labels)
    print(f"Test labels: {unique_test_labels}, counts: {[np.sum(test_labels == label) for label in unique_test_labels]}")
    
    try:
        auc_roc = roc_auc_score(test_labels, combined_test)
    except:
        auc_roc = float('nan')
    
    try:
        auc_pr = average_precision_score(test_labels, combined_test)
    except:
        auc_pr = float('nan')

    # 임계값 적용 Accuracy/F-beta
    preds_test = (combined_test > thr).astype(int)
    acc = accuracy_score(test_labels, preds_test)
    
    try:
        fbeta = fbeta_score(test_labels, preds_test, beta=beta)
    except:
        fbeta = float('nan')

    print(f"ROC-AUC (fusion) on Test: {auc_roc:.4f}")
    print(f"PR-AUC (fusion) on Test: {auc_pr:.4f}")
    print(f"Accuracy@thr (fusion) on Test: {acc:.4f}")
    print(f"F{beta}@thr (fusion) on Test: {fbeta:.4f}")
    
    # 공격 타입별 성능 분석
    print("\n=== Attack Type Breakdown ===")
    if 'attacker_type' in veremi_df.columns:
        # VeReMi 테스트셋에서 공격 타입별 성능 계산
        veremi_test_df = veremi_df.iloc[X_test.shape[0]:X_test.shape[0]*2] if len(veremi_df) > X_test.shape[0] else veremi_df
        
        # 간단한 공격 타입 분류 (0: Normal, 1+: Attack types)
        attack_types = np.unique(test_labels)
        for att_type in attack_types:
            mask = (test_labels == att_type)
            if np.sum(mask) > 10:  # 충분한 샘플이 있을 때만
                type_auc_roc = roc_auc_score(test_labels[mask], combined_test[mask]) if len(np.unique(test_labels[mask])) > 1 else float('nan')
                type_auc_pr = average_precision_score(test_labels[mask], combined_test[mask]) if len(np.unique(test_labels[mask])) > 1 else float('nan')
                type_acc = accuracy_score(test_labels[mask], preds_test[mask])
                type_name = "Normal" if att_type == 0 else f"Attack_{int(att_type)}"
                print(f"  {type_name} (n={np.sum(mask)}): ROC-AUC={type_auc_roc:.3f}, PR-AUC={type_auc_pr:.3f}, Acc={type_acc:.3f}")
    else:
        print("  Attack type breakdown not available (no attacker_type column)")

    # ----------------- 저장 -----------------
    os.makedirs(out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(out_dir, "model_lstm_combined.pth"))
    with open(os.path.join(out_dir, "training_meta_lstm_combined.json"), "w") as f:
        json.dump({
            "sequence_length": sequence_length,
            "input_dim": int(num_features),
            "alpha_fusion": float(alpha),
            "fbeta_beta": float(beta),
            "threshold_fusion": float(thr),
            "auc_roc_test_fusion": float(auc_roc) if not np.isnan(auc_roc) else None,
            "auc_pr_test_fusion": float(auc_pr) if not np.isnan(auc_pr) else None,
            "acc_test_fusion": float(acc),
            "fbeta_test_fusion": float(fbeta),
            "used_features": used_features,
            "training_data_info": {
                "v2aix_normal_sequences": int(X_v2aix.shape[0]),
                "veremi_normal_sequences": int(X_veremi_norm.shape[0]),
                "total_normal_sequences": int(X_normal_combined.shape[0]),
                "veremi_attack_sequences": int(X_veremi_attack.shape[0])
            }
        }, f, indent=2)

    plt.figure(figsize=(6,4))
    plt.plot(train_losses, label='Train Loss')
    if val_losses:
        plt.plot(val_losses, label='Val Loss')
    plt.title('Training Loss (Combined Normal Data)'); plt.xlabel('Epoch'); plt.ylabel('MSE Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "training_loss_lstm_combined.png"), dpi=200)
    plt.close()

if __name__ == "__main__":
    run_training()
