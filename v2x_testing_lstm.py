import os, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from v2x_preprocessing import V2XDataPreprocessor

# -------------------------------
# Model (LSTM-AutoEncoder) - training과 동일한 구조
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
        _, (hidden, cell) = self.encoder(x)
        decoder_input = hidden[-1].unsqueeze(1).repeat(1, x.size(1), 1)
        reconstructed, _ = self.decoder(decoder_input)
        return reconstructed

class V2XDataset(Dataset):
    def __init__(self, X, y): self.X = torch.FloatTensor(X); self.y = torch.FloatTensor(y)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

# -------------------------------
# Main Testing Function
# -------------------------------
def run_testing(
    artifacts_dir: str = "artifacts_lstm",
    v2aix_csv_path: str = "out/v2aix_preprocessed.csv",
    veremi_csv_path: str = "out/veremi_preprocessed.csv",
    random_state: int = 42,
    batch_size: int = 64
):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    with open(os.path.join(artifacts_dir, "training_meta_lstm.json"), "r") as f:
        meta = json.load(f)
    thr = float(meta["threshold"])
    input_dim = int(meta["input_dim"])
    seq_len = int(meta["sequence_length"])

    feature_columns = [
        'pos_x', 'pos_y', 'pos_z', 'spd_x', 'spd_y', 'spd_z',
        'heading', 'speed', 'acceleration', 'curvature'
    ]
    
    print("Loading preprocessed data from CSV...")
    v2aix_df = pd.read_csv(v2aix_csv_path)
    veremi_df = pd.read_csv(veremi_csv_path)

    pre = V2XDataPreprocessor(feature_columns=feature_columns)

    print("Creating sequences...")
    X_v2aix, y_v2aix = pre.create_sequences(v2aix_df, sequence_length=seq_len)
    X_veremi, y_veremi = pre.create_sequences(veremi_df, sequence_length=seq_len)
    X = np.concatenate([X_v2aix, X_veremi], axis=0)
    y = np.concatenate([y_v2aix, y_veremi], axis=0)

    _, X_temp, _, y_temp = train_test_split(X, y, test_size=0.4, random_state=random_state, stratify=y)
    _, X_test, _, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=random_state, stratify=y_temp)

    test_loader = DataLoader(V2XDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

    model = LSTMAutoEncoder(input_dim=input_dim, sequence_length=seq_len)
    model.load_state_dict(torch.load(os.path.join(artifacts_dir, "model_lstm.pth"), map_location=device))
    model.to(device)
    model.eval()

    preds, labels, errors = [], [], []
    with torch.no_grad():
        for batch, batch_labels in test_loader:
            batch = batch.to(device)
            rec = model(batch)
            err = torch.mean((batch - rec) ** 2, dim=[1, 2]).cpu().numpy()
            pred = (err > thr).astype(int)
            preds.extend(pred)
            labels.extend(batch_labels.numpy())
            errors.extend(err)

    preds, labels, errors = np.array(preds), np.array(labels), np.array(errors)
    
    print("\nANOMALY DETECTION RESULTS")
    print("="*40)
    print(f"Accuracy: {(preds == labels).mean():.4f}")
    print(f"AUC-ROC (errors): {roc_auc_score(labels, errors):.4f}")
    print("\nClassification Report:")
    print(classification_report(labels, preds, target_names=['Normal', 'Attack']))
    print("\nConfusion Matrix:")
    print(confusion_matrix(labels, preds))

if __name__ == "__main__":
    run_testing()