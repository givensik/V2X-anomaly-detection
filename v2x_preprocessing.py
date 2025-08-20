import json
import os
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from pathlib import Path
# -------------------------------
# Data Preprocessing & Dataset
# -------------------------------

class V2XDataPreprocessor:
    """V2X 데이터 전처리 클래스 - V2AIX와 VeReMi 데이터셋 모두에 사용"""
    def __init__(self, feature_columns: Optional[List[str]] = None):
        self.feature_columns = feature_columns or [
            'pos_x', 'pos_y', 'pos_z', 'spd_x', 'spd_y', 'spd_z',
            'heading', 'speed', 'acceleration', 'curvature'
        ]
        self.scaler = StandardScaler()
        self.is_fitted = False

    # ---------- V2AIX ----------
    def extract_cam_features_v2aix(self, json_data: Dict) -> List[Dict]:
        """V2AIX 데이터에서 CAM 메시지 특성 추출"""
        features = []
        if '/v2x/cam' in json_data:
            for cam_msg in json_data['/v2x/cam']:
                try:
                    cam = cam_msg['message']['cam']
                    cam_params = cam['cam_parameters']
                    basic_container = cam_params['basic_container']
                    ref_pos = basic_container['reference_position']

                    hf_container = cam_params['high_frequency_container']
                    if hf_container['choice'] == 0:
                        vehicle_hf = hf_container['basic_vehicle_container_high_frequency']
                        feature = {
                            'timestamp': cam_msg['recording_timestamp_nsec'],
                            'station_id': cam_msg['message']['header']['station_id']['value'],
                            'pos_x': ref_pos['longitude']['value'] / 1e7,
                            'pos_y': ref_pos['latitude']['value'] / 1e7,
                            'pos_z': ref_pos['altitude']['altitude_value']['value'] / 100,
                            'heading': vehicle_hf['heading']['heading_value']['value'] / 10,
                            'speed': vehicle_hf['speed']['speed_value']['value'] / 100,
                            'acceleration': vehicle_hf['longitudinal_acceleration']['longitudinal_acceleration_value']['value'] / 10,
                            'curvature': vehicle_hf['curvature']['curvature_value']['value'] / 10000,
                            'spd_x': 0.0, 'spd_y': 0.0, 'spd_z': 0.0
                        }
                        heading_rad = np.radians(feature['heading'])
                        feature['spd_x'] = feature['speed'] * np.cos(heading_rad)
                        feature['spd_y'] = feature['speed'] * np.sin(heading_rad)
                        features.append(feature)
                except (KeyError, TypeError):
                    continue
        return features

    # ---------- VeReMi ----------
    def extract_cam_features_veremi(self, json_data: List[Dict], ground_truth: List[Dict]) -> List[Dict]:
        """VeReMi 데이터에서 CAM 메시지 특성 추출 (GroundTruth와 매핑)"""
        features = []
        attacker_info = {}
        for gt_entry in ground_truth:
            if gt_entry.get('attackerType', 0) > 0:
                key = (gt_entry['time'], gt_entry['sender'], gt_entry['messageID'])
                attacker_info[key] = gt_entry['attackerType']


        for entry in json_data:
            if entry.get('type') == 3:
                try:
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
                        'heading': np.degrees(np.arctan2(entry['spd'][1], entry['spd'][0])),
                        'speed': float(np.sqrt(entry['spd'][0]**2 + entry['spd'][1]**2)),
                        'acceleration': 0.0,
                        'curvature': 0.0,
                        'is_attacker': is_attacker,
                        'attacker_type': attacker_type
                    }
                    features.append(feature)
                except (KeyError, TypeError):
                    continue
        return features

    # ---------- V2AIX Data Loading ----------
    def load_v2aix_data(self, data_path: str, max_files: int = 10) -> pd.DataFrame:
        all_features = []
        if os.path.isfile(data_path):
            json_files = [data_path]
        else:
            json_files = []
            for root, _, files in os.walk(data_path):
                for file in files:
                    if file.endswith('.json') and 'joined' not in file:
                        json_files.append(os.path.join(root, file))
                        if len(json_files) >= max_files:
                            break
                if len(json_files) >= max_files:
                    break

        for file_path in json_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                features = self.extract_cam_features_v2aix(data)
                all_features.extend(features)
            except Exception:
                continue

        df = pd.DataFrame(all_features)
        if not df.empty:
            df['dataset'] = 'v2aix'
            df['is_attacker'] = 0
            df['attacker_type'] = 0
        return df
    # ---------- VeReMi Data Loading ----------
    def load_veremi_data(self, json_log_path: str, ground_truth_path: str) -> pd.DataFrame:
        ground_truth = []
        with open(ground_truth_path, 'r') as f:
            for line in f:
                ground_truth.append(json.loads(line.strip()))
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
        if df.empty:
            return df
        numeric_features = [c for c in self.feature_columns if c in df.columns]
        df[numeric_features] = df[numeric_features].fillna(0)
        df[numeric_features] = df[numeric_features].replace([np.inf, -np.inf], 0)
        if not self.is_fitted:
            df[numeric_features] = self.scaler.fit_transform(df[numeric_features])
            self.is_fitted = True
        else:
            df[numeric_features] = self.scaler.transform(df[numeric_features])
        return df

    def create_sequences(self, df: pd.DataFrame, sequence_length: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        numeric_features = [c for c in self.feature_columns if c in df.columns]
        sequences, labels = [], []
        for station_id in df['station_id'].unique():
            station_data = df[df['station_id'] == station_id].sort_values('timestamp')
            if len(station_data) < sequence_length:
                continue
            for i in range(len(station_data) - sequence_length + 1):
                sequence = station_data[numeric_features].iloc[i:i+sequence_length].values
                label = station_data['is_attacker'].iloc[i:i+sequence_length].max()
                sequences.append(sequence)
                labels.append(label)
        return np.array(sequences), np.array(labels)

# -------- Helper to save/load preprocessor --------
def save_preprocessor(pre: V2XDataPreprocessor, path: str):
    import pickle
    with open(path, 'wb') as f:
        pickle.dump({'feature_columns': pre.feature_columns,
                     'scaler': pre.scaler,
                     'is_fitted': pre.is_fitted}, f)

def load_preprocessor(path: str) -> V2XDataPreprocessor:
    import pickle
    with open(path, 'rb') as f:
        data = pickle.load(f)
    pre = V2XDataPreprocessor(feature_columns=data['feature_columns'])
    pre.scaler = data['scaler']
    pre.is_fitted = data['is_fitted']
    return pre

# for VeReMi dataset 
def iter_veremi_pairs(roots):
    """
    roots: str 또는 str 리스트. 각 루트 아래를 재귀 탐색해서
           results/**/GroundTruthJSONlog.json 을 찾고,
           같은 폴더의 JSONlog-* 와 (log, gt) 페어를 yield.
    """
    if isinstance(roots, (str, Path)):
        roots = [roots]

    seen = set()
    for root in map(Path, roots):
        if not root.exists():
            continue
        # results 폴더만 타겟 (대/소문자 혼용 안전)
        for gt in root.rglob("results/GroundTruthJSONlog.json"):
            # 같은 디렉토리의 로그들(이름 변형 방지용으로 두 패턴 지원)
            candidates = list(gt.parent.glob("JSONlog-*.json")) + \
                         list(gt.parent.glob("*JSONlog*.json"))
            for log in candidates:
                if log.name == "GroundTruthJSONlog.json":
                    continue
                key = (str(log.resolve()), str(gt.resolve()))
                if key in seen:
                    continue
                seen.add(key)
                yield str(log), str(gt)


if __name__ == "__main__":
    preprocessor = V2XDataPreprocessor()
    
    # # V2AIX 데이터
    # v2aix_path = "V2AIX_Data/json/Mobile/V2X-only"
    # print(f"Loading V2AIX data from {v2aix_path}...")
    # v2aix_df = preprocessor.load_v2aix_data(v2aix_path, max_files=10000)

    # VeReMi 데이터: all/ 하위 모든 results 폴더에서 로그/GT 페어 탐색
    veremi_root = "VeReMi_Data/all"
    veremi_dfs = []
    save_interval = 100
    save_path = "out/veremi_temp.csv"
    processed_count = 0
    # 임시 파일에서 데이터 로드 (중단된 경우 재시작)
    if os.path.exists(save_path):
        try:
            temp_df = pd.read_csv(save_path)
            veremi_dfs.append(temp_df)
            processed_count = len(temp_df)
            print(f"Resuming VeReMi from {len(temp_df)} previously processed records.")
        except Exception as e:
            print(f"Error loading temp file, starting fresh: {e}")

    
    print(f"Loading all VeReMi logs and ground truths from {veremi_root}...")

    # iter_veremi_pairs 함수를 통해 모든 로그/GT 쌍을 자동으로 찾습니다.
    # 이미 처리된 파일은 건너뛰고 이어서 진행합니다.
    for log_path, gt_path in iter_veremi_pairs(veremi_root):
        if log_path in [df['_src'].iloc[0] for df in veremi_dfs if '_src' in df.columns]:
            continue

        print(f"Processing VeReMi log: {log_path}")
        df = preprocessor.load_veremi_data(log_path, gt_path)
        
        if not df.empty:
            df['_src'] = log_path
            veremi_dfs.append(df)
            processed_count += 1
            
            # 일정 간격마다 중간 저장
            if processed_count % save_interval == 0:
                current_df = pd.concat(veremi_dfs, ignore_index=True)
                current_df.to_csv(save_path, index=False)
                print(f"[{processed_count} pairs] Intermediate save to {save_path}")


    # # VeReMi 데이터셋의 로그와 Ground Truth 파일을 페어로 로드
    # for log_path, gt_path in iter_veremi_pairs(veremi_root):
    #     print(f"Loading VeReMi log: {log_path}")
    #     df = preprocessor.load_veremi_data(log_path, gt_path)

    #     if not df.empty:
    #         veremi_dfs.append(df)

    veremi_df = pd.concat(veremi_dfs, ignore_index=True) if veremi_dfs else pd.DataFrame()
    print(f"Total VeReMi records loaded: {len(veremi_df)}")
    
    # 각각 CSV로 저장
    print("Saving VeReMi preprocessed data...")
    # v2aix_df.to_csv("out/v2aix_preprocessed.csv", index=False)
    veremi_df.to_csv("out/veremi_preprocessed.csv", index=False)
    
    # (아래는 참고용: 전체 합쳐서 전처리/시퀀스 생성)
    # combined_df = pd.concat([v2aix_df, veremi_df], ignore_index=True)
    # processed_df = preprocessor.preprocess_features(combined_df)
    # sequences, labels = preprocessor.create_sequences(processed_df)
    # print(f"Sequences shape: {sequences.shape}, Labels shape: {labels.shape}")