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
from sklearn.metrics import roc_auc_score, average_precision_score, fbeta_score, accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

from v2x_preprocessing_lstm import V2XDataPreprocessor, fuse_scores  # ★ rule late-fusion 유틸 재사용
from sklearn.preprocessing import StandardScaler

# -------------------------------
# Models
# -------------------------------
class DenseAutoEncoder(nn.Module):
    """간단한 Dense AutoEncoder - 포인트와이즈 이상 탐지"""
    def __init__(self, input_dim, hidden_dim=16):
        super().__init__()
        # 더 간단하고 안전한 구조: 8 -> 16 -> 8 -> 16 -> 8
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),  # ReLU 대신 Tanh (더 안정적)
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh()
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.Tanh(), 
            nn.Linear(hidden_dim, input_dim)
            # 마지막 활성화 함수 제거 (선형 출력)
        )
    
    def forward(self, x):
        # x: (B, F) for pointwise
        # 입력 클리핑으로 안정성 확보
        x = torch.clamp(x, -10, 10)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

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
    criterion = nn.HuberLoss(reduction='none', delta=1.0)  # ★ Huber Loss (더 robust)
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

# -------------------------------
# Dense AutoEncoder Training
# -------------------------------
def train_dense_autoencoder(X_train, X_val=None, input_dim=8, epochs=100, lr=1e-3, 
                           batch_size=256, out_dir="artifacts_dense", device='cpu'):
    """Dense AutoEncoder 학습 (포인트와이즈)"""
    
    # 시퀀스 데이터를 포인트와이즈로 변환: (N, T, F) -> (N*T, F)
    if len(X_train.shape) == 3:
        X_train_flat = X_train.reshape(-1, X_train.shape[-1])
        print(f"Flattened training data: {X_train.shape} -> {X_train_flat.shape}")
    else:
        X_train_flat = X_train
    
    if X_val is not None and len(X_val.shape) == 3:
        X_val_flat = X_val.reshape(-1, X_val.shape[-1])
        print(f"Flattened validation data: {X_val.shape} -> {X_val_flat.shape}")
    else:
        X_val_flat = X_val
    
    # 데이터셋 및 로더 생성
    train_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(X_train_flat))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    if X_val_flat is not None:
        val_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(X_val_flat))
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    else:
        val_loader = None
    
    # 모델 및 최적화 (더 안정적인 설정)
    model = DenseAutoEncoder(input_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)  # weight decay 줄임
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.7)
    criterion = nn.MSELoss()
    
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    patience, bad_epochs = 15, 0
    
    print(f"Training Dense AutoEncoder for {epochs} epochs...")
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for batch, in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        val_loss = float('inf')
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch, in val_loader:
                    batch = batch.to(device)
                    recon = model(batch)
                    loss = criterion(recon, batch)
                    val_loss += loss.item()
            val_loss /= len(val_loader)
            val_losses.append(val_loss)
            
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                bad_epochs = 0
                os.makedirs(out_dir, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(out_dir, "best_dense_ae.pth"))
            else:
                bad_epochs += 1
        
        if epoch % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}]  train_loss={train_loss:.6f}  val_loss={val_loss:.6f}  lr={optimizer.param_groups[0]['lr']:.2e}")
        
        if bad_epochs >= patience:
            print("Early stopping!")
            break
    
    # Load best model
    if val_loader is not None:
        model.load_state_dict(torch.load(os.path.join(out_dir, "best_dense_ae.pth")))
        print("Loaded best model from early stopping")
    
    return model, train_losses, val_losses




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
def detailed_evaluation(y_true, y_scores, y_pred, model_name="Model"):
    """
    불균형 데이터에 적합한 상세 평가 지표 출력
    """
    print(f"\n=== {model_name} Detailed Evaluation ===")
    
    # 기본 지표들
    roc_auc = roc_auc_score(y_true, y_scores) if len(np.unique(y_true)) > 1 else float('nan')
    pr_auc = average_precision_score(y_true, y_scores) if len(np.unique(y_true)) > 1 else float('nan')
    
    # Classification 지표들
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    f15 = fbeta_score(y_true, y_pred, beta=1.5, zero_division=0)
    f2 = fbeta_score(y_true, y_pred, beta=2.0, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    
    # Confusion Matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # V2X 특화 지표
    attack_detection_rate = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Recall과 동일
    false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    
    # 랜덤 베이스라인과 비교
    random_baseline = np.mean(y_true)
    
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC: {pr_auc:.4f} (Random: {random_baseline:.4f})")
    print(f"Precision: {precision:.4f}")
    print(f"Recall (Attack Detection Rate): {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"F1.5-Score: {f15:.4f}")
    print(f"F2-Score: {f2:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"False Alarm Rate: {false_alarm_rate:.4f}")
    
    print(f"\nConfusion Matrix:")
    print(f"                Predicted")
    print(f"              Normal  Attack")
    print(f"Actual Normal   {tn:4d}    {fp:4d}")
    print(f"       Attack   {fn:4d}    {tp:4d}")
    
    # 성능 해석
    print(f"\n=== Performance Analysis ===")
    if pr_auc > random_baseline:
        print(f"✅ PR-AUC ({pr_auc:.4f}) > Random ({random_baseline:.4f})")
    else:
        print(f"❌ PR-AUC ({pr_auc:.4f}) ≤ Random ({random_baseline:.4f}) - Model is worse than random!")
    
    if false_alarm_rate < 0.1:
        print(f"✅ Low False Alarm Rate ({false_alarm_rate:.4f})")
    else:
        print(f"⚠️ High False Alarm Rate ({false_alarm_rate:.4f})")
    
    if attack_detection_rate > 0.7:
        print(f"✅ Good Attack Detection Rate ({attack_detection_rate:.4f})")
    elif attack_detection_rate > 0.5:
        print(f"⚠️ Moderate Attack Detection Rate ({attack_detection_rate:.4f})")
    else:
        print(f"❌ Poor Attack Detection Rate ({attack_detection_rate:.4f})")
    
    return {
        'roc_auc': roc_auc, 'pr_auc': pr_auc, 'precision': precision, 'recall': recall,
        'f1': f1, 'f15': f15, 'f2': f2, 'accuracy': accuracy,
        'attack_detection_rate': attack_detection_rate, 'false_alarm_rate': false_alarm_rate,
        'confusion_matrix': (tn, fp, fn, tp)
    }

def tune_alpha_and_threshold_pr_optimized(ae_errors_val, rule_seq_val, y_val, alphas=np.linspace(0,1,21)):
    """
    PR-AUC 최적화 기준으로 임계값 선택 (불균형 데이터에 더 적합)
    """
    best = {"alpha": 0.7, "thr": 0.0, "pr_auc": -1.0, "f1": -1.0, "f15": -1.0, "f2": -1.0}
    
    for alpha in alphas:
        comb = fuse_scores(ae_errors_val, rule_seq_val, alpha=alpha)
        
        # PR-AUC 계산
        pr_auc = average_precision_score(y_val, comb) if len(np.unique(y_val)) > 1 else 0.0
        
        # 다양한 F-beta 점수로 임계값 최적화
        thr_cands = np.linspace(comb.min(), comb.max(), 1000)
        best_f1, best_f15, best_f2 = -1, -1, -1
        best_thr_f1, best_thr_f15, best_thr_f2 = 0, 0, 0
        
        for thr in thr_cands:
            pred = (comb > thr).astype(int)
            if len(np.unique(pred)) > 1:  # 예측이 한 클래스만 나오지 않을 때
                f1 = f1_score(y_val, pred, zero_division=0)
                f15 = fbeta_score(y_val, pred, beta=1.5, zero_division=0)
                f2 = fbeta_score(y_val, pred, beta=2.0, zero_division=0)
                
                if f1 > best_f1:
                    best_f1, best_thr_f1 = f1, thr
                if f15 > best_f15:
                    best_f15, best_thr_f15 = f15, thr
                if f2 > best_f2:
                    best_f2, best_thr_f2 = f2, thr
        
        # PR-AUC가 가장 높은 alpha 선택
        if pr_auc > best["pr_auc"]:
            best.update({
                "alpha": alpha, "pr_auc": pr_auc,
                "f1": best_f1, "f15": best_f15, "f2": best_f2,
                "thr_f1": best_thr_f1, "thr_f15": best_thr_f15, "thr_f2": best_thr_f2
            })
    
    print(f"[PR-Optimized] Best alpha={best['alpha']:.3f}, PR-AUC={best['pr_auc']:.4f}")
    print(f"  F1-optimal threshold={best['thr_f1']:.6f} (F1={best['f1']:.4f})")
    print(f"  F1.5-optimal threshold={best['thr_f15']:.6f} (F1.5={best['f15']:.4f})")
    print(f"  F2-optimal threshold={best['thr_f2']:.6f} (F2={best['f2']:.4f})")
    
    return best

def tune_alpha_and_threshold(ae_errors_val, rule_seq_val, y_val, betas=(1.5,), alphas=np.linspace(0,1,21)):
    best = {"alpha":0.7, "beta":1.5, "thr":0.0, "fb":-1.0, "auc":-1.0}
    for alpha in alphas:
        comb = fuse_scores(ae_errors_val, rule_seq_val, alpha=alpha)
        # threshold는 comb 분포 위에서 탐색 (고해상도)
        thr_cands = np.linspace(comb.min(), comb.max(), 5000)
        for beta in betas:
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
    sequence_length: int = 4,         # ★ 20 → 8로 줄임 (더 짧은 패턴 학습)
    batch_size: int = 32,             # ★ 64 → 32로 줄임 (더 세밀한 학습)
    epochs: int = 70,                # ★ 50 → 100으로 늘림 (충분한 학습)
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
    
    # 시나리오 정보 확인
    print(f"V2AIX scenarios: {v2aix_df['scenario_id'].nunique()} unique scenarios")
    print(f"VeReMi scenarios: {veremi_df['scenario_id'].nunique()} unique scenarios")
    print(f"VeReMi attack distribution: {veremi_df['attacker_type'].value_counts().to_dict()}")

    # ★ 학습 피처는 전처리에서 사용한 Δ/파생/상대좌표 그대로 사용
    pre = V2XDataPreprocessor()  # feature_columns 기본값 사용 (dpos_x, dspeed, ..., rel_pos_y)
    used_features = [c for c in pre.feature_columns if c in v2aix_df.columns]
    print(f"Using features ({len(used_features)}): {used_features}")

    # ----------------- 시퀀스 생성 -----------------
    print("Creating sequences...")
    # V2AIX: 정상만으로 훈련
    X_v2aix, y_v2aix = pre.create_sequences(v2aix_df, sequence_length=sequence_length)
    print(f"V2AIX sequences: {X_v2aix.shape}, labels unique={np.unique(y_v2aix)}")

    # VeReMi: 검증/테스트 + rule_seq 함께 (스케일링 문제 해결)
    print("Fixing VeReMi scaling issues for LSTM...")
    # 1. 원본 데이터에서 극단값 클리핑
    veremi_features = veremi_df[used_features].values
    for i in range(veremi_features.shape[1]):
        q01, q99 = np.percentile(veremi_features[:, i], [1, 99])
        veremi_features[:, i] = np.clip(veremi_features[:, i], q01, q99)
    
    # 2. V2AIX 기준으로 재스케일링
    
    scaler = StandardScaler()
    v2aix_features = v2aix_df[used_features].values
    scaler.fit(v2aix_features)
    veremi_features_scaled = scaler.transform(veremi_features)
    
    # 3. 스케일링된 데이터로 DataFrame 업데이트
    veremi_df_fixed = veremi_df.copy()
    veremi_df_fixed[used_features] = veremi_features_scaled
    
    # 4. 시퀀스 생성
    X_veremi, y_veremi, rule_veremi = pre.create_sequences_with_rules(veremi_df_fixed, sequence_length=sequence_length)
    print(f"VeReMi sequences (fixed): {X_veremi.shape}, rule_seq shape={rule_veremi.shape}")
    
    print(f"V2AIX data range: [{X_v2aix.min():.3f}, {X_v2aix.max():.3f}]")
    print(f"VeReMi data range (fixed): [{X_veremi.min():.3f}, {X_veremi.max():.3f}]")

    # ----------------- 분할 -----------------
    # print(f"Train on PURE normal sequences (V2AIX only): {len(X_train_n)}")

    # VeReMi → 혼합 검증/테스트 1차 분할
    X_val_mix, X_test, y_val_mix, y_test, r_val_mix, r_test = train_test_split(
        X_veremi, y_veremi, rule_veremi,
        test_size=0.5, random_state=random_state, stratify=y_veremi
    )
    print(f"Validation MIX (VeReMi): {X_val_mix.shape}, Test MIX: {X_test.shape}")

    # ★ 학습용 validation을 V2AIX에서 분할 (동일 분포 보장)
    X_train_split, X_val_norm, y_train_split, y_val_norm = train_test_split(
        X_v2aix, y_v2aix, test_size=0.2, random_state=random_state
    )
    print(f"V2AIX split - Train: {X_train_split.shape}, Val: {X_val_norm.shape}")
    
    # 훈련 데이터 업데이트
    X_train_n = X_train_split
    y_train_n = y_train_split

    # ----------------- DataLoader -----------------
    train_loader = DataLoader(V2XDataset(X_train_n, y_train_n), batch_size=batch_size, shuffle=True)
    val_loader_norm  = DataLoader(V2XDataset(X_val_norm, y_val_norm), batch_size=batch_size, shuffle=False)  # 정상만
    val_loader_mix   = DataLoader(V2XDataset(X_val_mix, y_val_mix), batch_size=batch_size, shuffle=False)     # 혼합 (튜닝/진단)
    # val_loader   = DataLoader(V2XDataset(X_val, y_val), batch_size=batch_size, shuffle=False)  # ★순서보존
    test_loader  = DataLoader(V2XDataset(X_test, y_test), batch_size=batch_size, shuffle=False) # ★순서보존

    # ----------------- 모델 -----------------
    num_features = X_train_n.shape[2]
    
    # LSTM-AutoEncoder (개선된 학습 함수 사용)
    model = LSTMAutoEncoder(input_dim=num_features, sequence_length=sequence_length)

    print("Training LSTM-AutoEncoder with validation...")
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
    
    # 피처별 가중치 (움직임/변화에 더 민감하게)
    # feature order: ['dpos_x', 'dpos_y', 'dspeed', 'dheading_rad', 'acceleration', 'curvature', 'rel_pos_x', 'rel_pos_y']
    feature_weights = torch.tensor([1.2, 1.2, 1.5, 1.3, 1.2, 1.0, 1.0, 1.0], device=device)  # 움직임 피처에 가중
    
    with torch.no_grad():
        for batch, _ in val_loader_mix: # 혼합 검증셋
            batch = batch.to(device)
            recon = model(batch)  # AutoEncoder reconstruction
            # 피처별 가중 Huber Loss 계산 (delta=1.0)
            huber_err = torch.where(
                torch.abs(batch - recon) <= 1.0,
                0.5 * ((batch - recon) ** 2),
                1.0 * (torch.abs(batch - recon) - 0.5)
            )
            weighted_err = huber_err * feature_weights.view(1, 1, -1)
            err = torch.mean(weighted_err, dim=[1, 2])  # 가중 평균
            val_errors.extend(err.cpu().numpy())
    val_errors = np.array(val_errors)
    # 혼합 검증셋 라벨/룰 점수
    y_val = y_val_mix   
    r_val = r_val_mix

    # 진단 AUC (혼합셋 기준)
    ae_norm = (val_errors - np.percentile(val_errors, 1)) / (np.percentile(val_errors, 99) - np.percentile(val_errors, 1) + 1e-12)
    auc_ae_roc   = roc_auc_score(y_val, ae_norm) if len(np.unique(y_val))>1 else float('nan')
    auc_ae_pr    = average_precision_score(y_val, ae_norm) if len(np.unique(y_val))>1 else float('nan')
    auc_rule_roc = roc_auc_score(y_val, r_val)    if len(np.unique(y_val))>1 else float('nan')
    auc_rule_pr  = average_precision_score(y_val, r_val)    if len(np.unique(y_val))>1 else float('nan')
    print(f"[Diag] Val AUC (MIX) - AE: ROC={auc_ae_roc:.3f}, PR={auc_ae_pr:.3f}")
    print(f"[Diag] Val AUC (MIX) - Rule: ROC={auc_rule_roc:.3f}, PR={auc_rule_pr:.3f}")

    # PR-AUC 최적화 기준으로 임계값 선택 (불균형 데이터에 적합)
    alphas_combined = np.linspace(0, 1, 41)  # 0.0~1.0, 0.025 간격
    
    best_pr = tune_alpha_and_threshold_pr_optimized(val_errors, r_val, y_val, alphas=alphas_combined)
    alpha = best_pr["alpha"]
    
    # 다양한 임계값 옵션 제공
    thr_f1 = best_pr["thr_f1"]    # F1 최적화
    thr_f15 = best_pr["thr_f15"]  # F1.5 최적화 (Recall 중시)
    thr_f2 = best_pr["thr_f2"]    # F2 최적화 (Recall 더 중시)


    # ----------------- 테스트 -----------------
    print("Evaluating on test set (late-fusion)...")
    test_errors, test_labels = [], []
    with torch.no_grad():
        for batch, labels in test_loader:
            batch = batch.to(device)
            recon = model(batch)  # AutoEncoder reconstruction
            # 피처별 가중 Huber Loss 계산 (검증과 동일, delta=1.0)
            huber_err = torch.where(
                torch.abs(batch - recon) <= 1.0,
                0.5 * ((batch - recon) ** 2),
                1.0 * (torch.abs(batch - recon) - 0.5)
            )
            weighted_err = huber_err * feature_weights.view(1, 1, -1)
            err = torch.mean(weighted_err, dim=[1, 2])  # 가중 평균
            test_errors.extend(err.cpu().numpy())
            test_labels.extend(labels.numpy())
    test_errors = np.array(test_errors)
    test_labels = np.array(test_labels)

    # late-fusion
    combined_test = fuse_scores(test_errors, r_test, alpha=alpha)
    
    # 다양한 임계값으로 평가
    print("\n=== Test Results with Different Thresholds ===")
    
    # F1 최적 임계값
    preds_f1 = (combined_test > thr_f1).astype(int)
    metrics_f1 = detailed_evaluation(test_labels, combined_test, preds_f1, "F1-Optimized")
    
    # F1.5 최적 임계값 (Recall 중시)
    preds_f15 = (combined_test > thr_f15).astype(int)
    metrics_f15 = detailed_evaluation(test_labels, combined_test, preds_f15, "F1.5-Optimized (Recall Focus)")
    
    # F2 최적 임계값 (Recall 더 중시)
    preds_f2 = (combined_test > thr_f2).astype(int)
    metrics_f2 = detailed_evaluation(test_labels, combined_test, preds_f2, "F2-Optimized (High Recall)")
    
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
    torch.save(model.state_dict(), os.path.join(out_dir, "model_lstm.pth"))
    with open(os.path.join(out_dir, "training_meta_lstm.json"), "w") as f:
        json.dump({
            "sequence_length": sequence_length,
            "input_dim": int(num_features),
            "alpha_fusion": float(alpha),
            "fbeta_beta": float(beta),
            "threshold_fusion": float(thr),
            "auc_roc_test_fusion": float(auc_roc),
            "auc_pr_test_fusion": float(auc_pr),
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

def run_dense_training(
    v2aix_csv_path: str = "out/v2aix_preprocessed.csv",
    veremi_csv_path: str = "out/veremi_preprocessed.csv",
    out_dir: str = "artifacts_dense",
    epochs: int = 100,
    batch_size: int = 256,
    lr: float = 1e-4,  # 학습률 낮춤
    alpha: float = 0.7,
    beta: float = 1.5,
    random_state: int = 42,
):
    """Dense AutoEncoder 기반 학습"""
    os.makedirs(out_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    print("Loading preprocessed CSVs...")
    v2aix_df = pd.read_csv(v2aix_csv_path)
    veremi_df = pd.read_csv(veremi_csv_path)
    
    # 시나리오 정보 확인
    print(f"V2AIX scenarios: {v2aix_df['scenario_id'].nunique()} unique scenarios")
    print(f"VeReMi scenarios: {veremi_df['scenario_id'].nunique()} unique scenarios")
    print(f"VeReMi attack distribution: {veremi_df['attacker_type'].value_counts().to_dict()}")

    # 피처 선택
    pre = V2XDataPreprocessor()
    used_features = [c for c in pre.feature_columns if c in v2aix_df.columns]
    print(f"Using features ({len(used_features)}): {used_features}")

    # 포인트와이즈 데이터 준비 (시퀀스 생성 없이)
    X_v2aix = v2aix_df[used_features].values  # (N, F)
    X_veremi = veremi_df[used_features].values  # (M, F) 
    y_veremi = veremi_df['is_attacker'].values
    rule_veremi = veremi_df['rule_score'].values
    
    # 🔥 중요: VeReMi 데이터를 V2AIX와 같은 스케일로 정규화
    print("Re-scaling VeReMi data to match V2AIX distribution...")

    
    # 1단계: 극단값 클리핑 (VeReMi 원본에서)
    print("Clipping extreme values in VeReMi...")
    X_veremi_clipped = np.copy(X_veremi)
    for i in range(X_veremi.shape[1]):
        q01, q99 = np.percentile(X_veremi[:, i], [1, 99])
        X_veremi_clipped[:, i] = np.clip(X_veremi[:, i], q01, q99)
    
    # 2단계: V2AIX 기준으로 스케일링
    scaler = StandardScaler()
    scaler.fit(X_v2aix)  # V2AIX로 fit
    X_veremi = scaler.transform(X_veremi_clipped)  # 클리핑된 VeReMi에 적용
    
    print(f"After clipping & re-scaling - VeReMi mean: {X_veremi.mean(axis=0)}")
    print(f"After clipping & re-scaling - VeReMi std: {X_veremi.std(axis=0)}")
    
    print(f"V2AIX pointwise data: {X_v2aix.shape}")
    print(f"VeReMi pointwise data: {X_veremi.shape}")
    
    # VeReMi 분할
    X_val, X_test, y_val, y_test, r_val, r_test = train_test_split(
        X_veremi, y_veremi, rule_veremi,
        test_size=0.5, random_state=random_state, stratify=y_veremi
    )
    
    print(f"Training on V2AIX (normal): {X_v2aix.shape[0]} samples")
    print(f"Validation on VeReMi: {X_val.shape[0]} samples")
    print(f"Test on VeReMi: {X_test.shape[0]} samples")
    
    # Dense AutoEncoder 학습
    model, train_losses, val_losses = train_dense_autoencoder(
        X_v2aix, X_val, input_dim=len(used_features), 
        epochs=epochs, lr=lr, batch_size=batch_size, 
        out_dir=out_dir, device=device
    )
    
    # 평가
    print("\nEvaluating Dense AutoEncoder...")
    model.eval()
    with torch.no_grad():
        # Validation errors
        val_recon = model(torch.FloatTensor(X_val).to(device))
        val_errors = torch.mean((val_recon - torch.FloatTensor(X_val).to(device))**2, dim=1).cpu().numpy()
        
        # Test errors  
        test_recon = model(torch.FloatTensor(X_test).to(device))
        test_errors = torch.mean((test_recon - torch.FloatTensor(X_test).to(device))**2, dim=1).cpu().numpy()
    
    # AUC 계산
    val_auc_ae = roc_auc_score(y_val, val_errors) if len(np.unique(y_val)) > 1 else float('nan')
    val_auc_rule = roc_auc_score(y_val, r_val) if len(np.unique(y_val)) > 1 else float('nan')
    
    print(f"[Validation] AE ROC-AUC: {val_auc_ae:.4f}")
    print(f"[Validation] Rule ROC-AUC: {val_auc_rule:.4f}")
    
    # 임계값 튜닝 및 최종 평가
    best_params = tune_alpha_and_threshold(val_errors, r_val, y_val, betas=[beta], alphas=np.linspace(0, 1, 21))
    
    # 테스트 평가
    combined_test = fuse_scores(test_errors, r_test, alpha=best_params['alpha'])
    test_auc = roc_auc_score(y_test, combined_test) if len(np.unique(y_test)) > 1 else float('nan')
    test_preds = (combined_test > best_params['thr']).astype(int)
    test_acc = accuracy_score(y_test, test_preds)
    test_fbeta = fbeta_score(y_test, test_preds, beta=beta) if len(np.unique(y_test)) > 1 else float('nan')
    
    print(f"\n=== Final Dense AutoEncoder Results ===")
    print(f"Test ROC-AUC (fusion): {test_auc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test F{beta}: {test_fbeta:.4f}")
    
    # 결과 저장
    with open(os.path.join(out_dir, "dense_results.json"), "w") as f:
        json.dump({
            "val_auc_ae": float(val_auc_ae),
            "val_auc_rule": float(val_auc_rule),
            "test_auc_fusion": float(test_auc),
            "test_accuracy": float(test_acc),
            "test_fbeta": float(test_fbeta),
            "best_alpha": best_params['alpha'],
            "best_threshold": best_params['thr'],
            "used_features": used_features
        }, f, indent=2)
    
    return model, best_params

def run_domain_adaptation_training(
    v2aix_csv_path: str = "out/v2aix_preprocessed.csv",
    veremi_csv_path: str = "out/veremi_preprocessed.csv",
    out_dir: str = "artifacts_lstm",
    sequence_length: int = 4,
    batch_size: int = 32,
    pretrain_epochs: int = 50,        # ★ Pre-train 에폭
    finetune_epochs: int = 50,        # ★ Fine-tune 에폭
    pretrain_lr: float = 1e-3,        # ★ Pre-train 학습률
    finetune_lr: float = 5e-4,        # ★ Fine-tune 학습률 (더 낮게)
    hidden_dim: int = 32,
    alpha: float = 0.7,
    beta: float = 1.5,
    random_state: int = 42,
):
    """
    Domain Adaptation 학습:
    1. V2AIX(정상)로 Pre-train
    2. VeReMi(정상+공격)로 Fine-tune
    """
    os.makedirs(out_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    print("=== PHASE 1: Pre-training on V2AIX (Normal Only) ===")
    
    # 데이터 로딩 및 전처리
    print("Loading preprocessed CSVs...")
    v2aix_df = pd.read_csv(v2aix_csv_path)
    veremi_df = pd.read_csv(veremi_csv_path)
    
    print(f"V2AIX scenarios: {v2aix_df['scenario_id'].nunique()} unique scenarios")
    print(f"VeReMi scenarios: {veremi_df['scenario_id'].nunique()} unique scenarios")
    print(f"VeReMi attack distribution: {veremi_df['attacker_type'].value_counts().to_dict()}")

    # 특성 설정
    pre = V2XDataPreprocessor()
    used_features = [c for c in pre.feature_columns if c in v2aix_df.columns]
    print(f"Using features ({len(used_features)}): {used_features}")

    # V2AIX 시퀀스 생성 (Pre-training)
    print("Creating V2AIX sequences for pre-training...")
    X_v2aix, y_v2aix = pre.create_sequences(v2aix_df, sequence_length=sequence_length)
    print(f"V2AIX sequences: {X_v2aix.shape}, labels unique={np.unique(y_v2aix)}")
    
    # V2AIX 분할 (train/val)
    X_pretrain, X_preval, y_pretrain, y_preval = train_test_split(
        X_v2aix, y_v2aix, test_size=0.2, random_state=random_state
    )
    
    # 데이터 로더 생성
    pretrain_dataset = V2XDataset(X_pretrain, y_pretrain)
    preval_dataset = V2XDataset(X_preval, y_preval)
    pretrain_loader = DataLoader(pretrain_dataset, batch_size=batch_size, shuffle=True)
    preval_loader = DataLoader(preval_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Pre-train: {len(X_pretrain)} samples, Pre-val: {len(X_preval)} samples")

    # 모델 초기화
    input_dim = len(used_features)
    model = LSTMAutoEncoder(input_dim=input_dim, hidden_dim=hidden_dim, sequence_length=sequence_length)
    print(f"Model initialized: input_dim={input_dim}, hidden_dim={hidden_dim}, seq_len={sequence_length}")

    # Phase 1: Pre-training
    print(f"\n🔥 Starting Pre-training for {pretrain_epochs} epochs...")
    model, pretrain_losses, pretrain_val_losses = train_lstm_autoencoder(
        model, pretrain_loader, preval_loader, 
        epochs=pretrain_epochs, lr=pretrain_lr, device=device, out_dir=out_dir
    )
    
    # Pre-train 모델 저장
    pretrain_model_path = os.path.join(out_dir, "pretrained_model.pth")
    torch.save(model.state_dict(), pretrain_model_path)
    print(f"Pre-trained model saved to: {pretrain_model_path}")

    print("\n=== PHASE 2: Fine-tuning on VeReMi (Normal + Attack) ===")
    
    # VeReMi 데이터 스케일링 (이전과 동일)
    print("Fixing VeReMi scaling issues for fine-tuning...")
    veremi_features = veremi_df[used_features].values
    for i in range(veremi_features.shape[1]):
        q01, q99 = np.percentile(veremi_features[:, i], [1, 99])
        veremi_features[:, i] = np.clip(veremi_features[:, i], q01, q99)
    
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    v2aix_features = v2aix_df[used_features].values
    scaler.fit(v2aix_features)
    veremi_features_scaled = scaler.transform(veremi_features)
    
    veremi_df_fixed = veremi_df.copy()
    veremi_df_fixed[used_features] = veremi_features_scaled
    
    # VeReMi 시퀀스 생성 (Fine-tuning)
    X_veremi, y_veremi, rule_veremi = pre.create_sequences_with_rules(veremi_df_fixed, sequence_length=sequence_length)
    print(f"VeReMi sequences (fixed): {X_veremi.shape}, rule_seq shape={rule_veremi.shape}")
    
    # VeReMi 분할 (train/val/test)
    X_ft_temp, X_test, y_ft_temp, y_test, r_ft_temp, r_test = train_test_split(
        X_veremi, y_veremi, rule_veremi,
        test_size=0.4, random_state=random_state, stratify=y_veremi
    )
    
    X_finetune, X_val, y_finetune, y_val, r_finetune, r_val = train_test_split(
        X_ft_temp, y_ft_temp, r_ft_temp,
        test_size=0.5, random_state=random_state, stratify=y_ft_temp
    )
    
    print(f"Fine-tune: {len(X_finetune)} samples, Val: {len(X_val)} samples, Test: {len(X_test)} samples")
    print(f"Fine-tune attack ratio: {np.mean(y_finetune):.3f}")
    print(f"Val attack ratio: {np.mean(y_val):.3f}")
    print(f"Test attack ratio: {np.mean(y_test):.3f}")
    
    # 데이터 로더 생성
    finetune_dataset = V2XDataset(X_finetune, y_finetune)
    val_dataset = V2XDataset(X_val, y_val)
    test_dataset = V2XDataset(X_test, y_test)
    
    finetune_loader = DataLoader(finetune_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Phase 2: Fine-tuning (더 낮은 학습률)
    print(f"\n🔥 Starting Fine-tuning for {finetune_epochs} epochs with lr={finetune_lr}...")
    model, finetune_losses, finetune_val_losses = train_lstm_autoencoder(
        model, finetune_loader, val_loader, 
        epochs=finetune_epochs, lr=finetune_lr, device=device, out_dir=out_dir
    )
    
    # Fine-tuned 모델 저장
    finetuned_model_path = os.path.join(out_dir, "finetuned_model.pth")
    torch.save(model.state_dict(), finetuned_model_path)
    print(f"Fine-tuned model saved to: {finetuned_model_path}")

    # 평가 (기존 코드와 동일하지만 fine-tuned 데이터 사용)
    print("\n=== Evaluation on Fine-tuned Model ===")
    model.eval()
    val_errors = []
    
    # 피처별 가중치
    feature_weights = torch.tensor([1.2, 1.2, 1.5, 1.3, 1.2, 1.0, 1.0, 1.0], device=device)
    
    with torch.no_grad():
        for batch, _ in val_loader:
            batch = batch.to(device)
            recon = model(batch)
            # Huber Loss 계산
            huber_err = torch.where(
                torch.abs(batch - recon) <= 1.0,
                0.5 * ((batch - recon) ** 2),
                1.0 * (torch.abs(batch - recon) - 0.5)
            )
            weighted_err = huber_err * feature_weights.view(1, 1, -1)
            err = torch.mean(weighted_err, dim=[1, 2])
            val_errors.extend(err.cpu().numpy())
    
    val_errors = np.array(val_errors)
    
    # AUC 계산
    ae_norm = (val_errors - np.percentile(val_errors, 1)) / (np.percentile(val_errors, 99) - np.percentile(val_errors, 1) + 1e-12)
    auc_ae_roc = roc_auc_score(y_val, ae_norm) if len(np.unique(y_val)) > 1 else float('nan')
    auc_ae_pr = average_precision_score(y_val, ae_norm) if len(np.unique(y_val)) > 1 else float('nan')
    auc_rule_roc = roc_auc_score(y_val, r_val) if len(np.unique(y_val)) > 1 else float('nan')
    auc_rule_pr = average_precision_score(y_val, r_val) if len(np.unique(y_val)) > 1 else float('nan')
    
    # PR-AUC 최적화 기준으로 임계값 선택 (Domain Adaptation용)
    alphas_combined = np.linspace(0, 1, 41)  # 0.0~1.0, 0.025 간격
    best_pr = tune_alpha_and_threshold_pr_optimized(val_errors, r_val, y_val, alphas=alphas_combined)
    alpha = best_pr["alpha"]
    thr_f1 = best_pr["thr_f1"]
    thr_f15 = best_pr["thr_f15"]
    thr_f2 = best_pr["thr_f2"]
    
    # 검증셋에서 상세 평가
    val_combined = fuse_scores(val_errors, r_val, alpha=alpha)
    val_pred_f15 = (val_combined > thr_f15).astype(int)
    print(f"\n=== Validation Set Detailed Evaluation ===")
    detailed_evaluation(y_val, val_combined, val_pred_f15, "Fine-tuned Model (Validation)") 
    
    # 최종 테스트 (상세 평가)
    print("\n=== Final Test Results (Detailed) ===")
    test_errors = []
    with torch.no_grad():
        for batch, _ in test_loader:
            batch = batch.to(device)
            recon = model(batch)
            huber_err = torch.where(
                torch.abs(batch - recon) <= 1.0,
                0.5 * ((batch - recon) ** 2),
                1.0 * (torch.abs(batch - recon) - 0.5)
            )
            weighted_err = huber_err * feature_weights.view(1, 1, -1)
            err = torch.mean(weighted_err, dim=[1, 2])
            test_errors.extend(err.cpu().numpy())
    
    test_errors = np.array(test_errors)
    
    # Late fusion
    combined_test = fuse_scores(test_errors, r_test, alpha=alpha)
    
    # F1.5 최적 임계값으로 평가 (Recall 중시)
    test_pred_f15 = (combined_test > thr_f15).astype(int)
    final_metrics = detailed_evaluation(y_test, combined_test, test_pred_f15, "Final Test (F1.5-Optimized)")
    
    return model, {
        "pretrain": {"train_losses": pretrain_losses, "val_losses": pretrain_val_losses}, 
        "finetune": {"train_losses": finetune_losses, "val_losses": finetune_val_losses},
        "final_metrics": final_metrics,
        "best_params": best_pr
    }


if __name__ == "__main__":
    # 기존 단순 방법 테스트
    print("=== Testing LSTM AutoEncoder (No Domain Adaptation) ===")
    run_training()
    
    # Domain Adaptation 비교용 (주석 처리)
    # print("\n=== Domain Adaptation: V2AIX Pre-train → VeReMi Fine-tune ===")
    # run_domain_adaptation_training()
    
    # Dense AutoEncoder 비교용 (주석 처리)
    # print("\n=== Testing Dense AutoEncoder ===")
    # run_dense_training()
