# v2x_training_optimized.py
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
from sklearn.metrics import roc_auc_score, fbeta_score, accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt

from v2x_preprocessing_lstm import V2XDataPreprocessor, fuse_scores

# 기존 모델 클래스들 (동일)
class LSTMAutoEncoder(nn.Module):
    def __init__(self, input_dim, sequence_length, hidden_dim=128, n_layers=2, dropout=0.1):
        super().__init__()
        self.enc = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim,
                           num_layers=n_layers, batch_first=True,
                           dropout=dropout, bidirectional=True)
        self.bridge = nn.Linear(2*hidden_dim, hidden_dim)
        self.dec = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim,
                           num_layers=n_layers, batch_first=True, dropout=dropout)
        self.head = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        _, (h, c) = self.enc(x)
        h_last_fwd = h[-2,:,:]
        h_last_bwd = h[-1,:,:]
        z = torch.cat([h_last_fwd, h_last_bwd], dim=-1)
        z = torch.tanh(self.bridge(z))
        z_seq = z.unsqueeze(1).repeat(1, x.size(1), 1)
        y, _ = self.dec(z_seq)
        recon = self.head(y)
        return recon

class V2XDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

# 최적화된 학습 함수
def train_lstm_autoencoder(model, train_loader, val_loader=None, epochs=50, lr=1e-3, device='cpu', out_dir='artifacts_lstm_optimized'):
    criterion = nn.MSELoss(reduction='none')
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
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
            loss = criterion(recon, batch).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
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
            
            scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                bad_epochs = 0
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
    
    if val_loader is not None and os.path.exists(os.path.join(out_dir, "best_autoencoder.pth")):
        model.load_state_dict(torch.load(os.path.join(out_dir, "best_autoencoder.pth")))
        print("Loaded best model from early stopping")
    
    return model, train_losses, val_losses

# 최적화된 임계값 튜닝 함수
def tune_alpha_and_threshold_optimized(ae_errors_val, rule_seq_val, y_val, 
                                     alphas=[1.0], strategies=['balanced']):
    """최적화된 임계값 튜닝 - Balanced 전략 포함"""
    best = {"alpha": 1.0, "strategy": "balanced", "thr": 0.0, "score": -1.0, "auc": -1.0}
    
    for alpha in alphas:
        comb = fuse_scores(ae_errors_val, rule_seq_val, alpha=alpha)
        auc = roc_auc_score(y_val, comb) if len(np.unique(y_val)) > 1 else np.nan
        
        thr_cands = np.linspace(comb.min(), comb.max(), 5000)
        
        for strategy in strategies:
            best_score, best_thr = -1.0, comb.mean()
            
            for thr in thr_cands:
                preds = (comb > thr).astype(int)
                
                try:
                    if strategy == 'balanced':
                        # F1 × Accuracy 최적화
                        prec = precision_score(y_val, preds, zero_division=0)
                        rec = recall_score(y_val, preds, zero_division=0)
                        acc = accuracy_score(y_val, preds)
                        if prec + rec > 0:
                            f1 = 2 * prec * rec / (prec + rec)
                            score = f1 * acc
                        else:
                            score = 0
                    elif strategy == 'f1':
                        score = fbeta_score(y_val, preds, beta=1.0)
                    elif strategy == 'f1.5':
                        score = fbeta_score(y_val, preds, beta=1.5)
                    else:
                        score = fbeta_score(y_val, preds, beta=1.0)
                        
                    if score > best_score:
                        best_score, best_thr = score, thr
                except:
                    continue
                    
            if best_score > best["score"]:
                best = {
                    "alpha": float(alpha), 
                    "strategy": strategy, 
                    "thr": float(best_thr), 
                    "score": float(best_score), 
                    "auc": float(auc)
                }
    
    print(f"[OPTIMIZED] alpha={best['alpha']:.2f}, strategy={best['strategy']}, "
          f"thr={best['thr']:.6f}, score={best['score']:.4f}, AUC={best['auc']:.4f}")
    return best

# 최적화된 메인 학습 함수
def run_training_optimized(
    v2aix_csv_path: str = "out/v2aix_preprocessed.csv",
    veremi_csv_path: str = "out/veremi_preprocessed.csv",
    out_dir: str = "artifacts_lstm_optimized",
    sequence_length: int = 20,
    batch_size: int = 64,
    epochs: int = 50,
    lr: float = 1e-3,
    alpha: float = 1.0,               # ★ 최적화: AE만 사용
    strategy: str = "balanced",       # ★ 최적화: Balanced 전략
    random_state: int = 42,
):
    """최적화된 설정으로 학습 실행"""
    
    print("🚀 OPTIMIZED V2X TRAINING - Alpha=1.0, Balanced Strategy")
    print("=" * 60)
    
    os.makedirs(out_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("Loading preprocessed CSVs...")
    v2aix_df = pd.read_csv(v2aix_csv_path)
    veremi_df = pd.read_csv(veremi_csv_path)

    pre = V2XDataPreprocessor()
    used_features = [c for c in pre.feature_columns if c in v2aix_df.columns]
    print(f"Using features ({len(used_features)}): {used_features}")

    # 시퀀스 생성
    print("Creating sequences...")
    X_v2aix, y_v2aix = pre.create_sequences(v2aix_df, sequence_length=sequence_length)
    X_veremi, y_veremi, rule_veremi = pre.create_sequences_with_rules(veremi_df, sequence_length=sequence_length)
    
    # 스케일링
    print("Applying unified data scaling...")
    X_v2aix_flat = X_v2aix.reshape(-1, X_v2aix.shape[-1])
    X_veremi_flat = X_veremi.reshape(-1, X_veremi.shape[-1])
    X_combined = np.vstack([X_v2aix_flat, X_veremi_flat])
    
    pre.scaler.fit(X_combined)
    pre.is_fitted = True
    
    X_v2aix = pre.scaler.transform(X_v2aix_flat).reshape(X_v2aix.shape)
    X_veremi = pre.scaler.transform(X_veremi_flat).reshape(X_veremi.shape)

    # 분할
    X_val_mix, X_test, y_val_mix, y_test, r_val_mix, r_test = train_test_split(
        X_veremi, y_veremi, rule_veremi,
        test_size=0.5, random_state=random_state, stratify=y_veremi
    )
    
    X_train_split, X_val_norm, y_train_split, y_val_norm = train_test_split(
        X_v2aix, y_v2aix, test_size=0.2, random_state=random_state
    )
    
    X_train_n = X_train_split
    y_train_n = y_train_split

    # DataLoader
    train_loader = DataLoader(V2XDataset(X_train_n, y_train_n), batch_size=batch_size, shuffle=True)
    val_loader_norm = DataLoader(V2XDataset(X_val_norm, y_val_norm), batch_size=batch_size, shuffle=False)
    val_loader_mix = DataLoader(V2XDataset(X_val_mix, y_val_mix), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(V2XDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

    # 모델
    num_features = X_train_n.shape[2]
    model = LSTMAutoEncoder(input_dim=num_features, sequence_length=sequence_length)

    print("Training LSTM-AutoEncoder with optimized settings...")
    model, train_losses, val_losses = train_lstm_autoencoder(
        model, train_loader, val_loader_norm, epochs=epochs, lr=lr, device=device, out_dir=out_dir
    )

    # 검증
    print("Validating with optimized threshold selection...")
    model.eval()
    val_errors = []
    feature_weights = torch.tensor([1.2, 1.2, 1.5, 1.3, 1.2, 1.0, 1.0, 1.0], device=device)
    
    with torch.no_grad():
        for batch, _ in val_loader_mix:
            batch = batch.to(device)
            recon = model(batch)
            weighted_err = ((batch - recon) ** 2) * feature_weights.view(1, 1, -1)
            err = torch.mean(weighted_err, dim=[1, 2])
            val_errors.extend(err.cpu().numpy())
    
    val_errors = np.array(val_errors)
    y_val = y_val_mix
    r_val = r_val_mix

    # 최적화된 튜닝 (Alpha=1.0, Balanced 전략 우선)
    best = tune_alpha_and_threshold_optimized(
        val_errors, r_val, y_val, 
        alphas=[1.0],  # 최적값 고정
        strategies=['balanced']  # 최적 전략
    )
    
    optimized_alpha = best["alpha"]
    optimized_thr = best["thr"]
    optimized_strategy = best["strategy"]

    # 테스트
    print("Evaluating on test set with optimized settings...")
    test_errors, test_labels = [], []
    with torch.no_grad():
        for batch, labels in test_loader:
            batch = batch.to(device)
            recon = model(batch)
            weighted_err = ((batch - recon) ** 2) * feature_weights.view(1, 1, -1)
            err = torch.mean(weighted_err, dim=[1, 2])
            test_errors.extend(err.cpu().numpy())
            test_labels.extend(labels.numpy())
    
    test_errors = np.array(test_errors)
    test_labels = np.array(test_labels)

    # Late-fusion with optimized settings
    combined_test = fuse_scores(test_errors, r_test, alpha=optimized_alpha)
    auc = roc_auc_score(test_labels, combined_test) if len(np.unique(test_labels)) > 1 else float('nan')

    # 최적화된 임계값으로 평가
    preds_test = (combined_test > optimized_thr).astype(int)
    acc = accuracy_score(test_labels, preds_test)
    prec = precision_score(test_labels, preds_test)
    rec = recall_score(test_labels, preds_test)
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    balanced_score = f1 * acc

    print(f"\n�� OPTIMIZED TRAINING RESULTS:")
    print(f"Strategy: {optimized_strategy}")
    print(f"Alpha: {optimized_alpha}")
    print(f"Threshold: {optimized_thr:.6f}")
    print(f"AUC-ROC: {auc:.4f}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Balanced Score: {balanced_score:.4f}")

    # 저장
    torch.save(model.state_dict(), os.path.join(out_dir, "model_lstm_optimized.pth"))
    with open(os.path.join(out_dir, "training_meta_lstm_optimized.json"), "w") as f:
        json.dump({
            "sequence_length": sequence_length,
            "input_dim": int(num_features),
            "alpha_fusion": float(optimized_alpha),
            "strategy": optimized_strategy,
            "threshold_fusion": float(optimized_thr),
            "auc_test_fusion": float(auc),
            "acc_test_fusion": float(acc),
            "precision_test": float(prec),
            "recall_test": float(rec),
            "f1_test": float(f1),
            "balanced_score_test": float(balanced_score),
            "used_features": used_features
        }, f, indent=2)

    # 학습 곡선 저장
    plt.figure(figsize=(6,4))
    plt.plot(train_losses, label='Train Loss')
    if val_losses:
        plt.plot(val_losses, label='Val Loss')
    plt.title('Optimized Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "training_loss_optimized.png"), dpi=200)
    plt.close()
    
    return {
        'alpha': optimized_alpha,
        'strategy': optimized_strategy, 
        'threshold': optimized_thr,
        'auc': auc,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'balanced_score': balanced_score
    }

if __name__ == "__main__":
    results = run_training_optimized()
    print(f"\n✅ Optimized training completed!")
    print(f"Results saved in: artifacts_lstm_optimized/")