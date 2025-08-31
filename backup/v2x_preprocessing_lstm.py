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
            'dpos_x', 'dpos_y',          # 프레임 간 위치 변화 (단위 일치 이슈 제거)
            'dspeed', 'dheading_rad',    # 속도/헤딩 변화
            'acceleration',              # dv/dt
            'curvature',                  # yaw_rate/speed (안정식)
            'rel_pos_x', 'rel_pos_y'     # 상대 위치 (로컬 좌표계)
        ]
        self.scaler = StandardScaler()
        self.is_fitted = False

    # ---------- V2AIX ----------
    def extract_cam_features_v2aix(self, json_data: Dict, scenario_id: str = "") -> List[Dict]:
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
                            'heading': vehicle_hf['heading']['heading_value']['value'] / 10, # degree
                            'speed': vehicle_hf['speed']['speed_value']['value'] / 100, # m/s
                            'acceleration': vehicle_hf['longitudinal_acceleration']['longitudinal_acceleration_value']['value'] / 10, # m/s²
                            'curvature': vehicle_hf['curvature']['curvature_value']['value'] / 10000, # 1/m
                            'spd_x': 0.0, 'spd_y': 0.0, 'spd_z': 0.0,
                            'scenario_id': scenario_id
                        }
                        heading_rad = np.radians(feature['heading'])
                        feature['spd_x'] = feature['speed'] * np.cos(heading_rad)
                        feature['spd_y'] = feature['speed'] * np.sin(heading_rad)
                        features.append(feature)
                except (KeyError, TypeError):
                    continue
        return features

    # ---------- VeReMi ----------
    def extract_cam_features_veremi(self, json_data: List[Dict], ground_truth: List[Dict], scenario_id: str = "") -> List[Dict]:
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
                        'timestamp': entry['sendTime'], # s
                        'station_id': entry['sender'],
                        'pos_x': entry['pos'][0],
                        'pos_y': entry['pos'][1],
                        'pos_z': entry['pos'][2],
                        'spd_x': entry['spd'][0],
                        'spd_y': entry['spd'][1],
                        'spd_z': entry['spd'][2],
                        'heading': np.degrees(np.arctan2(entry['spd'][1], entry['spd'][0])),
                        'speed': float(np.sqrt(entry['spd'][0]**2 + entry['spd'][1]**2)),
                        'acceleration': 0.0, # 이후 Δ로 재계산하여 채움
                        'curvature': 0.0,   # 이후 yaw_rate/speed로 채움
                        'is_attacker': is_attacker,
                        'attacker_type': attacker_type,
                        'scenario_id': scenario_id
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
                # 파일 경로를 시나리오 ID로 사용
                scenario_id = file_path.replace('\\', '/').replace(data_path.replace('\\', '/'), '').strip('/')
                features = self.extract_cam_features_v2aix(data, scenario_id)
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
    def load_veremi_data(self, json_log_path: str, ground_truth_path: str, 
                        target_attacker_types: Optional[List[int]] = None) -> pd.DataFrame:
        """
        VeReMi 데이터 로딩 (특정 공격 타입 필터링 지원)
        
        Args:
            json_log_path: JSON 로그 파일 경로
            ground_truth_path: Ground Truth 파일 경로  
            target_attacker_types: 포함할 공격 타입 리스트 (None이면 모든 타입 포함)
                                 예: [0, 2] -> 정상(0)과 타입2 공격만 포함
        """
        ground_truth = []
        with open(ground_truth_path, 'r') as f:
            for line in f:
                ground_truth.append(json.loads(line.strip()))
        json_data = []
        with open(json_log_path, 'r') as f:
            for line in f:
                json_data.append(json.loads(line.strip()))
        
        # JSON 로그 파일 경로에서 시나리오 ID 추출 
        # 예: .../veins_maat.uc1.14505201.180205_165350/veins-maat/simulations/securecomm2018/results/JSONlog.json 
        # -> veins_maat.uc1.14505201.180205_165350
        scenario_path = Path(json_log_path)
        # results 디렉토리에서 최상위 시나리오 디렉토리까지 거슬러 올라가기
        current_path = scenario_path.parent  # results
        while current_path.name not in ['all', 'VeReMi_Data'] and current_path.parent != current_path:
            if current_path.name.startswith('veins_'):  # 시나리오 디렉토리 패턴
                scenario_id = current_path.name
                break
            current_path = current_path.parent
        else:
            scenario_id = scenario_path.parent.name  # 기본값
        features = self.extract_cam_features_veremi(json_data, ground_truth, scenario_id)
        df = pd.DataFrame(features)
        
        # VeReMi는 개별 파일에서 중복 제거하지 않고, 나중에 전체 데이터에서 중복 제거
        
        # 특정 공격 타입 필터링
        if not df.empty and target_attacker_types is not None:
            mask = df['attacker_type'].isin(target_attacker_types)
            df = df[mask].reset_index(drop=True)
            print(f"  Filtered to attacker types {target_attacker_types}: {len(df)} records")
        
        if not df.empty:
            df['dataset'] = 'veremi'
        return df
    


    def preprocess_features(self, df: pd.DataFrame, fit_on_mask: Optional[pd.Series] = None) -> pd.DataFrame:
        if df.empty:
            return df
        numeric_features = [c for c in self.feature_columns if c in df.columns]
        df[numeric_features] = df[numeric_features].fillna(0).replace([np.inf, -np.inf], 0)

        # fit_on_mask가 주어지면 그 부분으로만 fit (예: V2AIX 정상)
        if not self.is_fitted:
            fit_idx = fit_on_mask if fit_on_mask is not None else pd.Series(True, index=df.index)
            self.scaler.fit(df.loc[fit_idx, numeric_features].values)
            self.is_fitted = True

        df[numeric_features] = self.scaler.transform(df[numeric_features].values)
        return df


    def create_sequences(self, df: pd.DataFrame, sequence_length: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        numeric_features = [c for c in self.feature_columns if c in df.columns]
        sequences, labels = [], []

        # dataset, scenario_id, station_id까지 포함해서 그룹
        for (dataset, scenario_id, station_id), g in df.groupby(['dataset', 'scenario_id', 'station_id'], sort=False):
            station_data = g.sort_values('timestamp')
            if len(station_data) < sequence_length:
                continue
            X = station_data[numeric_features].values
            y = station_data['is_attacker'].values
            for i in range(len(station_data) - sequence_length + 1):
                sequences.append(X[i:i+sequence_length])
                labels.append(y[i:i+sequence_length].max())
        return np.array(sequences), np.array(labels)
    
    # 시퀀스 생성시 규칙 점수도 함께 반환
    def create_sequences_with_rules(self, df: pd.DataFrame, sequence_length: int = 10
                                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        AE 입력 시퀀스(X), 라벨(y), 규칙점수 시퀀스 집계(rule_seq_score)를 반환.
        rule_seq_score는 윈도우 내 rule_score의 '최댓값'.
        """
        numeric_features = [c for c in self.feature_columns if c in df.columns]
        sequences, labels, rule_scores = [], [], []

        for (dataset, scenario_id, station_id), g in df.groupby(['dataset', 'scenario_id', 'station_id'], sort=False):
            station_data = g.sort_values('timestamp')
            if len(station_data) < sequence_length:
                continue
            X = station_data[numeric_features].values
            y = station_data['is_attacker'].values
            r = station_data['rule_score'].values if 'rule_score' in station_data.columns else np.zeros(len(station_data))
            for i in range(len(station_data) - sequence_length + 1):
                sequences.append(X[i:i+sequence_length])
                labels.append(y[i:i+sequence_length].max())
                rule_scores.append(r[i:i+sequence_length].max())
        return np.array(sequences), np.array(labels), np.array(rule_scores)


    #   ------- Rule-based Scores -------
    def add_dynamic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Δ 기반 피처 계산(+ 상대 위치). 좌표를 로컬 미터로 통일해서 사용.
        """
        if df.empty:
            return df

        df = df.sort_values(['dataset','scenario_id','station_id','timestamp']).reset_index(drop=True)
        gkey = ['dataset','scenario_id','station_id']
        prev = df.groupby(gkey).shift(1)

        # --- (A) 데이터셋별 dt(초) ---
        # V2AIX: ns -> s, VeReMi: s 그대로
        is_v2aix = (df['dataset'] == 'v2aix')
        dt = pd.Series(0.0, index=df.index)
        dt.loc[is_v2aix] = (df.loc[is_v2aix,'timestamp'] - prev.loc[is_v2aix,'timestamp']) / 1e9
        dt.loc[~is_v2aix] = (df.loc[~is_v2aix,'timestamp'] - prev.loc[~is_v2aix,'timestamp'])
        dt = dt.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # --- (B) 로컬 미터 좌표 만들기 ---
        # V2AIX: pos_x(lon deg), pos_y(lat deg) / VeReMi: 이미 미터 좌표라고 가정
        pos_mx = pd.Series(0.0, index=df.index)
        pos_my = pd.Series(0.0, index=df.index)

        # 각 그룹의 첫 위치(원점)
        first_pos = df.groupby(gkey)[['pos_x','pos_y']].transform('first')

        # V2AIX 그룹: 경위도 -> 로컬 미터
        mask = is_v2aix
        if mask.any():
            mx, my = _latlon_to_local_xy(
                lat_deg=df.loc[mask,'pos_y'],
                lon_deg=df.loc[mask,'pos_x'],
                lat0_deg=first_pos.loc[mask,'pos_y'],
                lon0_deg=first_pos.loc[mask,'pos_x']
            )
            pos_mx.loc[mask] = mx.values
            pos_my.loc[mask] = my.values

        # VeReMi 그룹: 이미 미터로 가정 → 첫 위치를 (0,0)으로 평행이동
        mask = ~is_v2aix
        if mask.any():
            pos_mx.loc[mask] = (df.loc[mask,'pos_x'] - first_pos.loc[mask,'pos_x']).values
            pos_my.loc[mask] = (df.loc[mask,'pos_y'] - first_pos.loc[mask,'pos_y']).values

        df['pos_mx'] = pos_mx.fillna(0.0)
        df['pos_my'] = pos_my.fillna(0.0)

        # --- (C) Δ 기반 피처 (전부 미터 좌표에서) ---
        prev_mx = df.groupby(gkey)['pos_mx'].shift(1).fillna(df['pos_mx'])
        prev_my = df.groupby(gkey)['pos_my'].shift(1).fillna(df['pos_my'])
        dmx = (df['pos_mx'] - prev_mx).fillna(0.0)
        dmy = (df['pos_my'] - prev_my).fillna(0.0)
        df['dpos_x'] = dmx
        df['dpos_y'] = dmy

        dspeed = (df['speed'] - prev['speed']).fillna(0.0)
        df['dspeed'] = dspeed

        prev_h = prev['heading'].fillna(df['heading'])
        dh_deg = (df['heading'] - prev_h).fillna(0.0)
        dh_wrapped = ((dh_deg + 180) % 360) - 180
        dheading_rad = np.radians(dh_wrapped)
        df['dheading_rad'] = dheading_rad

        eps = 1e-6

        # 가속도 a ≈ dspeed / dt
        accel = dspeed / (dt + eps)
        df['acceleration'] = np.where(
            (df['acceleration'] == 0) | (~np.isfinite(df['acceleration'])),
            accel, df['acceleration']
        )

        # yaw_rate & curvature
        yaw_rate = dheading_rad / (dt + eps)     # rad/s
        df['__yaw_rate'] = yaw_rate
        curvature_stable = yaw_rate / (df['speed'].abs() + eps)
        df['curvature'] = np.where(
            (df['curvature'] == 0) | (~np.isfinite(df['curvature'])),
            curvature_stable, df['curvature']
        )

        # 상대 위치(미터 기준)
        first_mx = df.groupby(gkey)['pos_mx'].transform('first')
        first_my = df.groupby(gkey)['pos_my'].transform('first')
        df['rel_pos_x'] = (df['pos_mx'] - first_mx).fillna(0.0)
        df['rel_pos_y'] = (df['pos_my'] - first_my).fillna(0.0)

        for c in ['dpos_x','dpos_y','dspeed','dheading_rad','acceleration','curvature','__yaw_rate','rel_pos_x','rel_pos_y','pos_mx','pos_my']:
            df[c] = df[c].replace([np.inf, -np.inf], 0).fillna(0.0)

        return df
    def add_rule_scores(self, df: pd.DataFrame,
                        v_max: float = 40.0,      # 99% 지점 기준 (41.6)보다 약간 낮게
                        a_max: float = 6.0,       # 99% 지점 기준 (5.8)보다 약간 높게  
                        yaw_max: float = 0.5,     # 헤딩 변화율 유지
                        jump_slack: float = 2.0) -> pd.DataFrame:  # 점프 허용치 약간 완화
        if df.empty:
            df['rule_score'] = 0.0
            return df

        df = df.sort_values(['dataset','scenario_id','station_id','timestamp']).reset_index(drop=True)
        prev = df.groupby(['dataset','scenario_id','station_id']).shift(1)

        # 데이터셋별 dt(초)
        is_v2aix = (df['dataset'] == 'v2aix')
        dt = pd.Series(0.0, index=df.index)
        dt.loc[is_v2aix] = (df.loc[is_v2aix,'timestamp'] - prev.loc[is_v2aix,'timestamp']) / 1e9
        dt.loc[~is_v2aix] = (df.loc[~is_v2aix,'timestamp'] - prev.loc[~is_v2aix,'timestamp'])
        dt = dt.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        eps = 1e-6
        dt_safe = dt + eps

        # 1) 속도 상한
        v = df['speed'].abs()
        rule_speed = np.clip((v - v_max) / (v_max * 0.2 + eps), 0.0, 1.0)

        # 2) 가속도 상한
        a = df['acceleration'].abs()
        rule_accel = np.clip((a - a_max) / (a_max * 0.25 + eps), 0.0, 1.0)

        # 3) 헤딩 변화율 상한 (rad/s)
        yaw_rate = df['__yaw_rate'].abs()
        rule_yaw = np.clip((yaw_rate - yaw_max) / (yaw_max * 0.5 + eps), 0.0, 1.0)

        # 4) 점프(미터 기준): dpos는 이미 미터
        dmx = df['dpos_x']; dmy = df['dpos_y']
        jump_dist = np.sqrt(dmx*dmx + dmy*dmy)               # m
        expected = df['speed'].abs() * dt_safe + jump_slack   # m
        rule_jump = np.clip((jump_dist - expected) / (jump_slack + eps), 0.0, 1.0)

        w_speed = w_acc = w_yaw = w_jump = 0.25
        rule_score = (w_speed*rule_speed + w_acc*rule_accel + w_yaw*rule_yaw + w_jump*rule_jump)

        df['rule_speed'] = rule_speed
        df['rule_accel'] = rule_accel
        df['rule_yaw']   = rule_yaw
        df['rule_jump']  = rule_jump
        df['rule_score'] = rule_score.clip(0.0, 1.0)
        return df


 


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
def iter_veremi_pairs(roots, target_attacker_types=None):
    """
    roots: str 또는 str 리스트. 각 루트 아래를 재귀 탐색해서
           results/**/GroundTruthJSONlog.json 을 찾고,
           같은 폴더의 JSONlog-* 와 (log, gt) 페어를 yield.
    target_attacker_types: 특정 공격 타입의 시나리오만 선택 (예: [2]는 Type2만)
                          None이면 모든 타입 포함
    """
    if isinstance(roots, (str, Path)):
        roots = [roots]

    seen = set()
    for root in map(Path, roots):
        if not root.exists():
            continue
        # results 폴더만 타겟 (대/소문자 혼용 안전)
        for gt in root.rglob("results/GroundTruthJSONlog.json"):
            # 공격 타입 필터링: .sca 파일명으로 디렉토리 필터링
            if target_attacker_types is not None:
                sca_files = list(gt.parent.glob("AttackerType*.sca"))
                if sca_files:
                    # .sca 파일에서 공격 타입 추출
                    found_types = set()
                    for sca_file in sca_files:
                        # AttackerType2-start=3,0.1-#0.sca -> 2
                        import re
                        match = re.search(r'AttackerType(\d+)-', sca_file.name)
                        if match:
                            found_types.add(int(match.group(1)))
                    
                    # 원하는 공격 타입이 이 디렉토리에 없으면 스킵
                    if not any(att_type in found_types for att_type in target_attacker_types):
                        continue
                else:
                    # .sca 파일이 없는 디렉토리는 정상 시나리오로 간주
                    if 0 not in target_attacker_types:  # 정상 데이터를 원하지 않으면 스킵
                        continue
            
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

# -------- Late Fusion of AE and Rule Scores --------
def fuse_scores(ae_seq_scores: np.ndarray,
                rule_seq_scores: np.ndarray,
                alpha: float = 0.7,
                eps: float = 1e-12) -> np.ndarray:
    """
    late-fusion: combined = alpha * ae_norm + (1-alpha) * rule
      - ae_seq_scores: AE의 시퀀스 점수(재구성 오차 등)
      - rule_seq_scores: 위에서 만든 규칙 점수(0~1 권장)
      - alpha: AE 가중치 (0.7 권장)
    """
    # AE 점수를 0~1 정규화(95% 분위수 기준 - 정상 데이터 대부분을 낮은 점수로)
    ae = np.asarray(ae_seq_scores, dtype=float)
    q95 = np.percentile(ae, 95)  # 95% 분위수를 상한으로
    ae_min = ae.min()
    ae_norm = np.clip((ae - ae_min) / (q95 - ae_min + eps), 0.0, 1.0)  # 최솟값~95%를 0~1로

    rule = np.asarray(rule_seq_scores, dtype=float).clip(0.0, 1.0)
    combined = alpha * ae_norm + (1 - alpha) * rule
    return combined

# ------- Lat/Lon to Local XY (for VeReMi) -------
def _latlon_to_local_xy(lat_deg: pd.Series, lon_deg: pd.Series,
                        lat0_deg: pd.Series, lon0_deg: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """
    각 행마다 (lat0, lon0)를 원점으로 하는 근사적 로컬 ENU 변환.
    작은 영역 가정: dx = dlon * cos(lat0) * 111320 (m), dy = dlat * 111320 (m)
    """
    # 차이(deg)
    dlat = lat_deg - lat0_deg
    dlon = lon_deg - lon0_deg

    # m/deg
    m_per_deg_lat = 111320.0
    # lat0는 deg -> rad 변환 후 cos
    m_per_deg_lon = 111320.0 * np.cos(np.radians(lat0_deg.clip(-89.999, 89.999)))

    x = dlon * m_per_deg_lon
    y = dlat * m_per_deg_lat
    return x, y

if __name__ == "__main__":
    # 1. 전처리기 및 파일 제한 설정
    preprocessor = V2XDataPreprocessor()
    max_veremi_files = 1000  # VeReMi 파일 처리 개수 증가 (300→1000)
    
    # VeReMi 공격 타입 설정 (원하는 타입들을 선택)
    # 
    # 🎯 디렉토리 필터링: 특정 공격 타입의 시나리오가 있는 폴더에서만 데이터 로드
    # 🔍 데이터 필터링: 로드된 데이터에서 특정 공격 타입만 선택
    #
    # 사용 예시:
    # None: 모든 타입 포함 (기본값)
    # [0]: 정상 데이터만 (정상 시나리오 디렉토리에서만)
    # [2]: 타입2 공격만 (AttackerType2*.sca가 있는 디렉토리에서만)
    # [0, 2]: 정상 + 타입2 공격 (정상 시나리오 + AttackerType2 시나리오 디렉토리)
    # [1, 2, 4]: 타입1, 2, 4 공격 (해당 공격 타입 시나리오 디렉토리들에서만)
    
    directory_filter_types = [1, 2, 4]  # Type 1, 2, 4 공격 시나리오 디렉토리에서 로드
    data_filter_types = [0, 1, 2, 4]    # 로드된 데이터에서 정상 + Type 1, 2, 4 선택
    
    print(f"Directory filtering for attacker types: {directory_filter_types}")
    print(f"Data filtering for attacker types: {data_filter_types}")

    # --- 2. V2AIX와 VeReMi 데이터를 각각 로드 ---
    v2aix_path = "V2AIX_Data/json/Mobile/V2X-only"
    print(f"Loading V2AIX data from {v2aix_path}...")
    v2aix_df = preprocessor.load_v2aix_data(v2aix_path, max_files=1000000)

    veremi_root = "VeReMi_Data/all"
    veremi_dfs = []
    print(f"\nLoading VeReMi logs (limit: {max_veremi_files} files)...")
    print(f"Only loading from directories with attack types: {directory_filter_types}")
    
    all_pairs = list(iter_veremi_pairs(veremi_root, target_attacker_types=directory_filter_types))
    print(f"Found {len(all_pairs)} file pairs in filtered directories")
    
    for i, (log_path, gt_path) in enumerate(all_pairs):
        if i >= max_veremi_files:
            break
        file_name = log_path.split('/')[-1] if '/' in log_path else log_path.split('\\')[-1]
        print(f"Processing file {i+1}/{min(len(all_pairs), max_veremi_files)}: {file_name}")
        df = preprocessor.load_veremi_data(log_path, gt_path, data_filter_types)
        if not df.empty:
            veremi_dfs.append(df)

    veremi_df = pd.concat(veremi_dfs, ignore_index=True) if veremi_dfs else pd.DataFrame()

    # VeReMi 중복 메시지 제거 (전체 데이터에서)
    if not veremi_df.empty:
        print(f"Before deduplication: {len(veremi_df)} records")
        # sendTime, sender, scenario_id 조합으로 중복 제거 (같은 시나리오에서 같은 sender가 같은 시간에 보낸 메시지)
        veremi_df = veremi_df.drop_duplicates(subset=['timestamp', 'station_id', 'scenario_id'], keep='first').reset_index(drop=True)
        print(f"After deduplication: {len(veremi_df)} records")

    # VeReMi 공격 타입별 통계 출력
    if not veremi_df.empty:
        print("="*100)
        print(f"\nVeReMi data loaded: {len(veremi_df)} records")
        attacker_stats = veremi_df['attacker_type'].value_counts().sort_index()
        print("Attacker type distribution:")
        for att_type, count in attacker_stats.items():
            att_name = "Normal" if att_type == 0 else f"Attack Type {att_type}"
            print(f"  {att_name}: {count} records ({count/len(veremi_df)*100:.1f}%)")
    
    # --- 3. 두 데이터프레임을 하나로 합치기 ---
    print("\nCombining V2AIX and VeReMi datasets...")
    combined_df = pd.concat([v2aix_df, veremi_df], ignore_index=True)
    print(f"Total records combined: {len(combined_df)}")
    
    # 전체 데이터셋의 공격/정상 비율 출력
    if not combined_df.empty:
        total_attackers = len(combined_df[combined_df['is_attacker'] == 1])
        total_normal = len(combined_df[combined_df['is_attacker'] == 0])
        print(f"Overall - Normal: {total_normal}, Attackers: {total_attackers}")
        if total_attackers > 0:
            print(f"Attack ratio: {total_attackers/(total_normal+total_attackers)*100:.2f}%")

    # --- 4. 동적 피처 추가 ---
    print("Adding dynamic features to the combined dataset...")
    combined_df = preprocessor.add_dynamic_features(combined_df)
    # after: combined_df = preprocessor.add_dynamic_features(combined_df)
    print("V2AIX dt/dpos stats:",
        combined_df.loc[combined_df.dataset=="v2aix", ["dpos_x","dpos_y"]].abs().mean().to_dict())

    print("VeReMi dt/dpos stats:",
        combined_df.loc[combined_df.dataset=="veremi", ["dpos_x","dpos_y"]].abs().mean().to_dict())

    # dt, yaw_rate, curvature sanity
    for name, mask in [("V2AIX", combined_df.dataset=="v2aix"),
                    ("VeReMi", combined_df.dataset=="veremi")]:
        dt_guess = combined_df.loc[mask].groupby(["dataset","station_id"])["timestamp"].diff()
        if name=="V2AIX":
            dt_guess = dt_guess/1e9  # ns -> s
        print(name, "dt(s) mean=", float(dt_guess.replace([np.inf,-np.inf], np.nan).dropna().mean()))
        print(name, "yaw_rate mean=", float(combined_df.loc[mask,"__yaw_rate"].abs().mean()))
        print(name, "curvature mean=", float(combined_df.loc[mask,"curvature"].abs().mean()))
    combined_df = preprocessor.add_rule_scores(combined_df)  # ★ 규칙 점수 추가
    
    print("Dynamic features and rule scores added.")

    # --- 5. 스케일링: V2AIX 정상 데이터로만 fit  (★중요: fit 1회만!)
    fit_mask = (combined_df['dataset'] == 'v2aix') & (combined_df['is_attacker'] == 0)
    print("Preprocessing (scaling) the combined dataset with V2AIX normals as fit set...")
    processed_df = preprocessor.preprocess_features(combined_df, fit_on_mask=fit_mask)

    # 시퀀스 만들기 (AE 입력 + 라벨 + 규칙점수집계)
    X, y, rule_seq = preprocessor.create_sequences_with_rules(processed_df, sequence_length=20)

    # --- 6. 최종 전처리된 데이터를 다시 분리하여 저장 ---
    print("Splitting and saving the final preprocessed data...")
    final_v2aix_df = processed_df[processed_df['dataset'] == 'v2aix']
    final_veremi_df = processed_df[processed_df['dataset'] == 'veremi']

    os.makedirs("out", exist_ok=True)
    final_v2aix_df.to_csv("out/v2aix_preprocessed.csv", index=False)
    final_veremi_df.to_csv("out/veremi_preprocessed.csv", index=False)

    print("\nPreprocessing finished successfully.")
    print(f"V2AIX data saved: {len(final_v2aix_df)} rows")
    print(f"VeReMi data saved: {len(final_veremi_df)} rows")

    