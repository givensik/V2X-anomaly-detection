import json
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class V2XDataPreprocessor:
    """V2X 데이터 전처리 클래스 - V2AIX와 VeReMi 데이터셋 모두에 사용"""
    
    def __init__(self, feature_columns: Optional[List[str]] = None):
        """
        Args:
            feature_columns: 사용할 특성 컬럼 리스트. None이면 기본 특성 사용
        """
        self.feature_columns = feature_columns or [
            'pos_x', 'pos_y', 'pos_z', 'spd_x', 'spd_y', 'spd_z',
            'heading', 'speed', 'acceleration', 'curvature'
        ]
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def extract_cam_features_v2aix(self, json_data: Dict) -> List[Dict]:
        """V2AIX 데이터에서 CAM 메시지 특성 추출"""
        features = []
        
        if '/v2x/cam' in json_data:
            for cam_msg in json_data['/v2x/cam']:
                try:
                    cam = cam_msg['message']['cam']
                    cam_params = cam['cam_parameters']
                    
                    # 기본 컨테이너에서 위치 정보 추출
                    basic_container = cam_params['basic_container']
                    ref_pos = basic_container['reference_position']
                    
                    # 고주파 컨테이너에서 속도, 방향 등 추출
                    hf_container = cam_params['high_frequency_container']
                    if hf_container['choice'] == 0:  # basic_vehicle_container_high_frequency
                        vehicle_hf = hf_container['basic_vehicle_container_high_frequency']
                        
                        feature = {
                            'timestamp': cam_msg['recording_timestamp_nsec'],
                            'station_id': cam_msg['message']['header']['station_id']['value'],
                            'pos_x': ref_pos['longitude']['value'] / 1e7,  # 스케일 조정
                            'pos_y': ref_pos['latitude']['value'] / 1e7,
                            'pos_z': ref_pos['altitude']['altitude_value']['value'] / 100,  # cm to m
                            'heading': vehicle_hf['heading']['heading_value']['value'] / 10,  # 0.1도 단위
                            'speed': vehicle_hf['speed']['speed_value']['value'] / 100,  # 0.01 m/s 단위
                            'acceleration': vehicle_hf['longitudinal_acceleration']['longitudinal_acceleration_value']['value'] / 10,  # 0.1 m/s²
                            'curvature': vehicle_hf['curvature']['curvature_value']['value'] / 10000,  # 스케일 조정
                            'spd_x': 0.0,  # CAM에는 개별 속도 성분이 없으므로 0으로 설정
                            'spd_y': 0.0,
                            'spd_z': 0.0
                        }
                        
                        # 속도 성분 계산 (heading과 speed를 이용)
                        heading_rad = np.radians(feature['heading'])
                        feature['spd_x'] = feature['speed'] * np.cos(heading_rad)
                        feature['spd_y'] = feature['speed'] * np.sin(heading_rad)
                        
                        features.append(feature)
                        
                except (KeyError, TypeError) as e:
                    continue
                    
        return features
    
    def extract_cam_features_veremi(self, json_data: Dict, ground_truth: Dict) -> List[Dict]:
        """VeReMi 데이터에서 CAM 메시지 특성 추출 (GroundTruth와 매핑)"""
        features = []
        
        # GroundTruth에서 공격자 정보 추출
        attacker_info = {}
        for gt_entry in ground_truth:
            if gt_entry.get('attackerType', 0) > 0:  # 공격자
                key = (gt_entry['time'], gt_entry['sender'], gt_entry['messageID'])
                attacker_info[key] = gt_entry['attackerType']
        
        # JSON 로그에서 CAM 메시지 추출 (type 3이 CAM 메시지)
        for entry in json_data:
            if entry.get('type') == 3:  # CAM 메시지
                try:
                    # GroundTruth와 매핑
                    key = (entry['sendTime'], entry['sender'], entry['messageID'])
                    is_attacker = key in attacker_info
                    attacker_type = attacker_info.get(key, 0)
                    
                    feature = {
                        'timestamp': entry['sendTime'],
                        'station_id': entry['sender'],
                        'pos_x': entry['pos'][0],
                        'pos_y': entry['pos'][1], 
                        'pos_z': entry['pos'][2],
                        'spd_x': entry['spd'][0],
                        'spd_y': entry['spd'][1],
                        'spd_z': entry['spd'][2],
                        'heading': np.arctan2(entry['spd'][1], entry['spd'][0]) * 180 / np.pi,
                        'speed': np.sqrt(entry['spd'][0]**2 + entry['spd'][1]**2),
                        'acceleration': 0.0,  # VeReMi에는 가속도 정보가 없음
                        'curvature': 0.0,     # VeReMi에는 곡률 정보가 없음
                        'is_attacker': is_attacker,
                        'attacker_type': attacker_type
                    }
                    
                    features.append(feature)
                    
                except (KeyError, TypeError) as e:
                    continue
                    
        return features
    
    def load_v2aix_data(self, data_path: str, max_files: int = 10) -> pd.DataFrame:
        """V2AIX 데이터 로드 및 전처리"""
        all_features = []
        
        # 데이터 경로에서 JSON 파일들 찾기
        if os.path.isfile(data_path):
            json_files = [data_path]
        else:
            json_files = []
            for root, dirs, files in os.walk(data_path):
                for file in files:
                    if file.endswith('.json') and 'joined' not in file:
                        json_files.append(os.path.join(root, file))
                        if len(json_files) >= max_files:
                            break
                if len(json_files) >= max_files:
                    break
        
        print(f"Loading {len(json_files)} V2AIX files...")
        
        for file_path in json_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                features = self.extract_cam_features_v2aix(data)
                all_features.extend(features)
                
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue
        
        df = pd.DataFrame(all_features)
        if not df.empty:
            df['dataset'] = 'v2aix'
            df['is_attacker'] = 0  # V2AIX는 정상 데이터
            df['attacker_type'] = 0
            
        return df
    
    def load_veremi_data(self, json_log_path: str, ground_truth_path: str) -> pd.DataFrame:
        """VeReMi 데이터 로드 및 전처리"""
        # GroundTruth 로드
        ground_truth = []
        with open(ground_truth_path, 'r') as f:
            for line in f:
                ground_truth.append(json.loads(line.strip()))
        
        # JSON 로그 로드
        json_data = []
        with open(json_log_path, 'r') as f:
            for line in f:
                json_data.append(json.loads(line.strip()))
        
        features = self.extract_cam_features_veremi(json_data, ground_truth)
        df = pd.DataFrame(features)
        
        if not df.empty:
            df['dataset'] = 'veremi'
            
        return df
    
    def preprocess_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """특성 전처리 및 정규화"""
        if df.empty:
            return df
        
        # 수치형 특성만 선택
        numeric_features = [col for col in self.feature_columns if col in df.columns]
        
        # 결측값 처리
        df[numeric_features] = df[numeric_features].fillna(0)
        
        # 무한값 처리
        df[numeric_features] = df[numeric_features].replace([np.inf, -np.inf], 0)
        
        # 특성 정규화
        if not self.is_fitted:
            df[numeric_features] = self.scaler.fit_transform(df[numeric_features])
            self.is_fitted = True
        else:
            df[numeric_features] = self.scaler.transform(df[numeric_features])
        
        return df
    
    def create_sequences(self, df: pd.DataFrame, sequence_length: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """시계열 시퀀스 생성"""
        numeric_features = [col for col in self.feature_columns if col in df.columns]
        
        sequences = []
        labels = []
        
        for station_id in df['station_id'].unique():
            station_data = df[df['station_id'] == station_id].sort_values('timestamp')
            
            if len(station_data) < sequence_length:
                continue
                
            for i in range(len(station_data) - sequence_length + 1):
                sequence = station_data[numeric_features].iloc[i:i+sequence_length].values
                label = station_data['is_attacker'].iloc[i:i+sequence_length].max()  # 시퀀스 내 공격자 여부
                
                sequences.append(sequence)
                labels.append(label)
        
        return np.array(sequences), np.array(labels)

class AutoEncoder(nn.Module):
    """AutoEncoder 모델"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [64, 32, 16]):
        super(AutoEncoder, self).__init__()
        
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder
        decoder_layers = []
        hidden_dims_reversed = hidden_dims[::-1]
        
        for i, hidden_dim in enumerate(hidden_dims_reversed):
            if i == len(hidden_dims_reversed) - 1:
                decoder_layers.extend([
                    nn.Linear(prev_dim, input_dim),
                    nn.Tanh()  # 출력을 -1~1 범위로 제한
                ])
            else:
                decoder_layers.extend([
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.2)
                ])
            prev_dim = hidden_dim
        
        self.decoder = nn.Sequential(*decoder_layers)
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x):
        return self.encoder(x)

class V2XDataset(Dataset):
    """V2X 데이터셋"""
    
    def __init__(self, sequences: np.ndarray, labels: np.ndarray):
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.FloatTensor(labels)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

class AnomalyDetector:
    """이상탐지 시스템"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [64, 32, 16]):
        self.model = AutoEncoder(input_dim, hidden_dims)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.threshold = None
        
    def train(self, train_loader: DataLoader, epochs: int = 100, device: str = 'cpu'):
        """모델 학습"""
        self.model.train()
        self.model.to(device)
        
        train_losses = []
        
        for epoch in range(epochs):
            epoch_loss = 0
            for batch_sequences, _ in train_loader:
                batch_sequences = batch_sequences.to(device)
                
                # 시퀀스를 평면화
                batch_size, seq_len, features = batch_sequences.shape
                flattened = batch_sequences.view(batch_size, -1)
                
                # Forward pass
                reconstructed = self.model(flattened)
                loss = self.criterion(reconstructed, flattened)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
            
            train_losses.append(epoch_loss / len(train_loader))
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {train_losses[-1]:.6f}')
        
        return train_losses
    
    def compute_threshold(self, val_loader: DataLoader, percentile: float = 95, device: str = 'cpu'):
        """재구성 오차 임계값 계산"""
        self.model.eval()
        reconstruction_errors = []
        
        with torch.no_grad():
            for batch_sequences, _ in val_loader:
                batch_sequences = batch_sequences.to(device)
                
                batch_size, seq_len, features = batch_sequences.shape
                flattened = batch_sequences.view(batch_size, -1)
                
                reconstructed = self.model(flattened)
                errors = torch.mean((flattened - reconstructed) ** 2, dim=1)
                reconstruction_errors.extend(errors.cpu().numpy())
        
        self.threshold = np.percentile(reconstruction_errors, percentile)
        print(f"Threshold (95th percentile): {self.threshold:.6f}")
        
        return reconstruction_errors
    
    def predict(self, test_loader: DataLoader, device: str = 'cpu') -> Tuple[np.ndarray, np.ndarray]:
        """이상 탐지 예측"""
        self.model.eval()
        predictions = []
        true_labels = []
        reconstruction_errors = []
        
        with torch.no_grad():
            for batch_sequences, batch_labels in test_loader:
                batch_sequences = batch_sequences.to(device)
                
                batch_size, seq_len, features = batch_sequences.shape
                flattened = batch_sequences.view(batch_size, -1)
                
                reconstructed = self.model(flattened)
                errors = torch.mean((flattened - reconstructed) ** 2, dim=1)
                
                # 임계값을 기준으로 이상 탐지
                batch_predictions = (errors > self.threshold).cpu().numpy()
                
                predictions.extend(batch_predictions)
                true_labels.extend(batch_labels.cpu().numpy())
                reconstruction_errors.extend(errors.cpu().numpy())
        
        return np.array(predictions), np.array(true_labels), np.array(reconstruction_errors)

def main():
    """메인 실행 함수"""
    print("V2X Anomaly Detection System")
    print("=" * 50)
    
    # 데이터 경로 설정
    v2aix_path = "V2AIX_Data/json/Mobile/V2X-only/Aachen"
    veremi_json_path = "VeReMi_Data/results/JSONlog-0-7-A0.json"
    veremi_ground_truth_path = "VeReMi_Data/results/GroundTruthJSONlog.json"
    
    # 전처리기 초기화
    preprocessor = V2XDataPreprocessor()
    
    # V2AIX 데이터 로드 (정상 데이터)
    print("Loading V2AIX data (normal)...")
    v2aix_df = preprocessor.load_v2aix_data(v2aix_path, max_files=5)
    
    # VeReMi 데이터 로드 (공격 데이터 포함)
    print("Loading VeReMi data (with attacks)...")
    veremi_df = preprocessor.load_veremi_data(veremi_json_path, veremi_ground_truth_path)
    
    # 데이터 결합
    combined_df = pd.concat([v2aix_df, veremi_df], ignore_index=True)
    print(f"Combined dataset shape: {combined_df.shape}")
    print(f"Normal samples: {len(combined_df[combined_df['is_attacker'] == 0])}")
    print(f"Attack samples: {len(combined_df[combined_df['is_attacker'] == 1])}")
    
    # 특성 전처리
    combined_df = preprocessor.preprocess_features(combined_df)
    
    # 시퀀스 생성
    print("Creating sequences...")
    sequences, labels = preprocessor.create_sequences(combined_df, sequence_length=10)
    print(f"Sequences shape: {sequences.shape}")
    print(f"Labels shape: {labels.shape}")
    
    # 데이터 분할
    X_train, X_temp, y_train, y_temp = train_test_split(
        sequences, labels, test_size=0.4, random_state=42, stratify=labels
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    # 정상 데이터만으로 학습 데이터 구성
    normal_indices = np.where(y_train == 0)[0]
    X_train_normal = X_train[normal_indices]
    y_train_normal = y_train[normal_indices]
    
    print(f"Training on normal data: {len(X_train_normal)} sequences")
    
    # 데이터셋 및 데이터로더 생성
    train_dataset = V2XDataset(X_train_normal, y_train_normal)
    val_dataset = V2XDataset(X_val, y_val)
    test_dataset = V2XDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # AutoEncoder 모델 초기화
    input_dim = sequences.shape[1] * sequences.shape[2]  # sequence_length * features
    detector = AnomalyDetector(input_dim, hidden_dims=[128, 64, 32])
    
    # 모델 학습
    print("Training AutoEncoder...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_losses = detector.train(train_loader, epochs=50, device=device)
    
    # 임계값 계산
    print("Computing threshold...")
    val_errors = detector.compute_threshold(val_loader, percentile=95, device=device)
    
    # 테스트 데이터로 평가
    print("Evaluating on test data...")
    predictions, true_labels, test_errors = detector.predict(test_loader, device=device)
    
    # 결과 출력
    print("\n" + "=" * 50)
    print("ANOMALY DETECTION RESULTS")
    print("=" * 50)
    
    print(f"Accuracy: {(predictions == true_labels).mean():.4f}")
    print(f"AUC-ROC: {roc_auc_score(true_labels, test_errors):.4f}")
    
    print("\nClassification Report:")
    print(classification_report(true_labels, predictions, target_names=['Normal', 'Attack']))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(true_labels, predictions)
    print(cm)
    
    # 결과 시각화
    plt.figure(figsize=(15, 5))
    
    # 학습 손실
    plt.subplot(1, 3, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    # 재구성 오차 분포
    plt.subplot(1, 3, 2)
    normal_errors = test_errors[true_labels == 0]
    attack_errors = test_errors[true_labels == 1]
    
    plt.hist(normal_errors, bins=50, alpha=0.7, label='Normal', density=True)
    plt.hist(attack_errors, bins=50, alpha=0.7, label='Attack', density=True)
    plt.axvline(detector.threshold, color='red', linestyle='--', label=f'Threshold: {detector.threshold:.4f}')
    plt.title('Reconstruction Error Distribution')
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Density')
    plt.legend()
    
    # ROC 곡선
    plt.subplot(1, 3, 3)
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(true_labels, test_errors)
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title(f'ROC Curve (AUC = {roc_auc_score(true_labels, test_errors):.4f})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    
    plt.tight_layout()
    plt.savefig('anomaly_detection_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nResults saved to 'anomaly_detection_results.png'")

if __name__ == "__main__":
    main()
