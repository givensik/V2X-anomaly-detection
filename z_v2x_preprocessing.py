# v2x_preprocessing.py

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import os

WINDOW_SIZE = 5
# FEATURES = ["pos_x", "pos_y", "pos_z", "speed", "heading", "curvature"]
FEATURES = ["pos_x", "pos_y", "pos_z", "speed", "heading", "curvature", "dpos_x", "dpos_y"]

def fit_and_save_scaler(df: pd.DataFrame, feature_cols: list, scaler_path: str):
    X = df[feature_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, scaler_path)
    return X_scaled, scaler


def load_and_transform_scaler(df: pd.DataFrame, feature_cols: list, scaler_path: str):
    scaler = joblib.load(scaler_path)
    X = df[feature_cols].values
    X_scaled = scaler.transform(X)
    return X_scaled


def make_sequences(df: pd.DataFrame, X_scaled: np.ndarray, window_size: int, label_col: str = None):
    sequences = []
    labels = []
    for (station_id, scenario_id), group in df.groupby(["station_id", "scenario_id"]):
        group = group.sort_values("timestamp")
        values = X_scaled[group.index]
        for i in range(len(values) - window_size + 1):
            sequences.append(values[i:i + window_size])
            if label_col:
                label_seq = group[label_col].values[i:i + window_size]
                labels.append(int(np.mean(label_seq) > 0.5))
            else:
                labels.append(0)  # 정상으로 간주
    return np.array(sequences), np.array(labels)

# 예측 기반 시퀀스 생성 함수
def make_prediction_sequences(df: pd.DataFrame, X_scaled: np.ndarray, window_size: int, pred_len: int, label_col: str = None):
    X_seq, Y_seq, labels = [], [], []
    for (station_id, scenario_id), group in df.groupby(["station_id", "scenario_id"]):
        group = group.sort_values("timestamp")
        values = X_scaled[group.index]
        if label_col:
            label_values = group[label_col].values
        for i in range(len(values) - window_size - pred_len + 1):
            X_seq.append(values[i:i + window_size])
            Y_seq.append(values[i + window_size:i + window_size + pred_len])
            if label_col:
                labels.append(int(np.mean(label_values[i + window_size:i + window_size + pred_len]) > 0.5))
            else:
                labels.append(0)
    return np.array(X_seq), np.array(Y_seq), np.array(labels)

if __name__ == "__main__":
    os.makedirs("z_data/", exist_ok=True)

    # V2AIX 전처리 (학습용)
    df_v2aix = pd.read_csv("z_test/v2aix_etsi_standardized.csv")
    df_v2aix["dpos_x"] = df_v2aix.groupby(["station_id", "scenario_id"])["pos_x"].diff().fillna(0)
    df_v2aix["dpos_y"] = df_v2aix.groupby(["station_id", "scenario_id"])["pos_y"].diff().fillna(0)
    
    X_scaled_v2aix, _ = fit_and_save_scaler(df_v2aix, FEATURES, "z_data/scaler_v2aix.pkl")
    # X_seq_v2aix, y_seq_v2aix = make_sequences(df_v2aix, X_scaled_v2aix, WINDOW_SIZE) # 기존 시퀀스 생성
    # np.save("z_data/train_X.npy", X_seq_v2aix) 
    # np.save("z_data/train_y.npy", y_seq_v2aix)
    
    # prediction 시퀀스 생성
    X_seq_v2aix, Y_seq_v2aix, _ = make_prediction_sequences(df_v2aix, X_scaled_v2aix, window_size=WINDOW_SIZE, pred_len=1)
    np.save("z_data/train_X.npy", X_seq_v2aix)
    np.save("z_data/train_Y.npy", Y_seq_v2aix)

    # VeReMi 전처리 (테스트용)
    df_veremi = pd.read_csv("z_test/veremi_etsi_standardized.csv")
    df_veremi["heading"] = (df_veremi["heading"] + 360) % 360
    # 기존 AE 시퀀스 생성
    # X_scaled_veremi = load_and_transform_scaler(df_veremi, FEATURES, "z_data/scaler_v2aix.pkl")
    # X_seq_veremi, y_seq_veremi = make_sequences(df_veremi, X_scaled_veremi, WINDOW_SIZE, label_col="is_attacker")
    # np.save("z_data/test_X.npy", X_seq_veremi)
    # np.save("z_data/test_y.npy", y_seq_veremi)
    df_veremi["dpos_x"] = df_veremi.groupby(["station_id", "scenario_id"])["pos_x"].diff().fillna(0)
    df_veremi["dpos_y"] = df_veremi.groupby(["station_id", "scenario_id"])["pos_y"].diff().fillna(0)

    # prediction 시퀀스 생성
    X_scaled_veremi = load_and_transform_scaler(df_veremi, FEATURES, "z_data/scaler_v2aix.pkl")
    X_seq_veremi, Y_seq_veremi, labels_veremi = make_prediction_sequences(
        df_veremi, X_scaled_veremi, window_size=WINDOW_SIZE, pred_len=1, label_col="is_attacker"
    )
    np.save("z_data/test_X.npy", X_seq_veremi)
    np.save("z_data/test_Y.npy", Y_seq_veremi)
    np.save("z_data/test_labels.npy", labels_veremi)

