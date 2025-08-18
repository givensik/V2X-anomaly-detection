import os, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

from v2x_preprocessing import V2XDataPreprocessor, load_preprocessor

class AutoEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dims=[128,64,32]):
        super().__init__()
        enc, prev = [], input_dim
        for h in hidden_dims:
            enc += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(0.2)]; prev = h
        self.encoder = nn.Sequential(*enc)
        dec, rev = [], list(hidden_dims)[::-1]
        for i, h in enumerate(rev):
            if i == len(rev)-1: dec += [nn.Linear(prev, input_dim), nn.Tanh()]
            else: dec += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(0.2)]
            prev = h
        self.decoder = nn.Sequential(*dec)
    def forward(self, x):
        z = self.encoder(x); return self.decoder(z)

class V2XDataset(Dataset):
    def __init__(self, X, y): self.X = torch.FloatTensor(X); self.y = torch.FloatTensor(y)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

def run_testing(
    artifacts_dir: str,
    v2aix_csv_path: str,
    veremi_csv_path: str,
    sequence_length: int = None,
    random_state: int = 42,
    batch_size: int = 32
):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Load artifacts
    with open(os.path.join(artifacts_dir, "training_meta.json"), "r") as f:
        meta = json.load(f)
    thr = float(meta["threshold"])
    input_dim = int(meta["input_dim"])
    seq_len = int(sequence_length or meta["sequence_length"])

    pre = load_preprocessor(os.path.join(artifacts_dir, "preprocessor.pkl"))
    # CSV 데이터 불러오기
    v2aix_df = pd.read_csv(v2aix_csv_path)
    veremi_df = pd.read_csv(veremi_csv_path)
    df = pd.concat([v2aix_df, veremi_df], ignore_index=True)
    df = pre.preprocess_features(df)
    X, y = pre.create_sequences(df, sequence_length=seq_len)

    # Same split strategy
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=random_state, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=random_state, stratify=y_temp)

    test_loader = DataLoader(V2XDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

    # Load model
    model = AutoEncoder(input_dim)
    model.load_state_dict(torch.load(os.path.join(artifacts_dir, "model.pth"), map_location=device))
    model.to(device); model.eval()

    preds, labels, errors = [], [], []
    with torch.no_grad():
        for batch, batch_labels in test_loader:
            b,s,f = batch.shape
            flat = batch.view(b,-1).to(device)
            rec = model(flat)
            err = torch.mean((flat - rec) ** 2, dim=1).cpu().numpy()
            pred = (err > thr).astype(int)
            preds.extend(pred); labels.extend(batch_labels.numpy()); errors.extend(err)

    preds = np.array(preds); labels = np.array(labels); errors = np.array(errors)
    acc = float((preds == labels).mean())
    auc = float(roc_auc_score(labels, errors)) if len(set(labels)) > 1 else float('nan')

    print("ANOMALY DETECTION RESULTS")
    print("="*40)
    print(f"Accuracy: {acc:.4f}")
    print(f"AUC-ROC (errors): {auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(labels, preds, target_names=['Normal','Attack']))
    print("Confusion Matrix:")
    print(confusion_matrix(labels, preds))

    # Plots
    out_png = os.path.join(artifacts_dir, "test_results.png")
    plt.figure(figsize=(15,5))
    # Error hist
    plt.subplot(1,3,1)
    plt.hist(errors[labels==0], bins=50, alpha=0.7, label='Normal', density=True)
    plt.hist(errors[labels==1], bins=50, alpha=0.7, label='Attack', density=True)
    plt.axvline(thr, linestyle='--', label=f'Threshold={thr:.4f}')
    plt.title('Reconstruction Error Dist.'); plt.legend()
    # ROC
    fpr, tpr, _ = roc_curve(labels, errors)
    plt.subplot(1,3,2)
    plt.plot(fpr, tpr); plt.plot([0,1],[0,1],'k--')
    plt.title(f'ROC (AUC={auc:.4f})')
    # Empty pane for prettier spacing or future metrics
    plt.subplot(1,3,3)
    plt.axis('off'); plt.text(0.1, 0.6, f'Accuracy: {acc:.4f}', fontsize=12)
    plt.text(0.1, 0.4, f'Threshold: {thr:.6f}', fontsize=12)
    plt.tight_layout(); plt.savefig(out_png, dpi=200, bbox_inches='tight'); plt.close()
    return {"accuracy": acc, "auc": auc, "figure": out_png}

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Test V2X anomaly detector")
    ap.add_argument("--artifacts_dir", type=str, required=True)
    ap.add_argument("--v2aix_csv_path", type=str, required=True)
    ap.add_argument("--veremi_csv_path", type=str, required=True)
    ap.add_argument("--sequence_length", type=int, default=None)
    ap.add_argument("--random_state", type=int, default=42)
    ap.add_argument("--batch_size", type=int, default=32)
    args = ap.parse_args()
    run_testing(**vars(args))
