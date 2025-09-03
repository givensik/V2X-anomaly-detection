import json
import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

class V2XDataLoader:
    """V2X 데이터 로딩 클래스 - 간단하게 정리

    - V2AIX: 위도/경도 → 미터 좌표로 변환
    - VeReMi: 원본 미터 좌표 그대로 사용
    """
    
    def __init__(self):
        """데이터 로더 초기화"""
        self.standardize_units = True
        self.ETSI_UNITS = {
            'position': 'meter',
            'speed': 'm/s', 
            'heading': 'degree',
            'timestamp': 'second'
        }
    
    # ---------- V2AIX 데이터 로딩 ----------
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
                        # ETSI CAM 표준에서 실제 단위로 변환
                        lon_deg = ref_pos['longitude']['value'] / 1e7
                        lat_deg = ref_pos['latitude']['value'] / 1e7
                        alt_m = ref_pos['altitude']['altitude_value']['value'] / 100
                        
                        # 시나리오별 첫 번째 위치를 기준으로 로컬 미터 좌표로 변환
                        # 일단 위도/경도를 저장하고 나중에 변환
                        pos_x = lon_deg  # 경도 (degree)
                        pos_y = lat_deg  # 위도 (degree)
                        
                        feature = {
                            'timestamp': cam_msg['recording_timestamp_nsec'] / 1e9,  # ns → s 변환
                            'station_id': cam_msg['message']['header']['station_id']['value'],
                            'pos_x': pos_x,
                            'pos_y': pos_y,
                            'pos_z': alt_m,
                            'heading': vehicle_hf['heading']['heading_value']['value'] / 10,  # degree
                            'speed': vehicle_hf['speed']['speed_value']['value'] / 100,  # m/s
                            'acceleration': vehicle_hf['longitudinal_acceleration']['longitudinal_acceleration_value']['value'] / 10,  # m/s²
                            'curvature': vehicle_hf['curvature']['curvature_value']['value'] / 10000,  # 1/m
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

    def load_v2aix_data(self, data_path: str, max_files: int = 10) -> pd.DataFrame:
        """V2AIX 데이터 로딩"""
        all_features = []
        
        # 단일 파일인지 디렉토리인지 확인
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

        print(f"V2AIX 파일 {len(json_files)}개 처리 중...")
        
        for i, file_path in enumerate(json_files):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 파일 경로를 시나리오 ID로 사용
                scenario_id = file_path.replace('\\', '/').replace(data_path.replace('\\', '/'), '').strip('/')
                features = self.extract_cam_features_v2aix(data, scenario_id)
                all_features.extend(features)
                
                if (i + 1) % 100 == 0:
                    print(f"  처리 완료: {i + 1}/{len(json_files)} 파일")
                    
            except Exception as e:
                print(f"  파일 처리 실패: {file_path} - {str(e)}")
                continue

        df = pd.DataFrame(all_features)
        if not df.empty:
            df['dataset'] = 'v2aix'
            df['is_attacker'] = 0  # V2AIX는 모두 정상 데이터
            df['attacker_type'] = 0
            
            # 미터 좌표 변환 적용
            df = self._convert_v2aix_to_meters(df)
            
        print(f"V2AIX 데이터 로딩 완료: {len(df)} 레코드")
        return df

    def _convert_v2aix_to_meters(self, df: pd.DataFrame) -> pd.DataFrame:
        """V2AIX 위도/경도를 시나리오별 로컬 미터 좌표로 변환"""
        if df.empty:
            return df
            
        print("V2AIX 좌표를 미터 단위로 변환 중...")
        
        # 원본 좌표 백업
        df = df.copy()
        df['pos_x_original'] = df['pos_x']  # 경도
        df['pos_y_original'] = df['pos_y']  # 위도
        
        # 시나리오별로 정렬
        df = df.sort_values(['scenario_id', 'station_id', 'timestamp']).reset_index(drop=True)
        
        # 각 시나리오의 첫 번째 위치를 기준점으로 설정
        first_positions = df.groupby('scenario_id')[['pos_x', 'pos_y']].first().reset_index()
        first_positions.columns = ['scenario_id', 'first_lon', 'first_lat']
        
        # 기준점 정보 병합
        df = df.merge(first_positions, on='scenario_id', how='left')
        
        # 좌표 변환 계수
        METERS_PER_DEGREE_LAT = 111320.0  # 위도 1도 ≈ 111.32km
        
        # 위도/경도 차이 계산
        dlat = df['pos_y'] - df['first_lat']  # 위도 차이
        dlon = df['pos_x'] - df['first_lon']  # 경도 차이
        
        # 경도는 위도에 따라 달라짐 (cos(lat) 적용)
        lat_rad = np.radians(df['first_lat'].clip(-89.999, 89.999))
        meters_per_degree_lon = METERS_PER_DEGREE_LAT * np.cos(lat_rad)
        
        # 미터 단위로 변환
        df['pos_x'] = dlon * meters_per_degree_lon  # X: 동-서 방향 (경도)
        df['pos_y'] = dlat * METERS_PER_DEGREE_LAT  # Y: 남-북 방향 (위도)
        
        # 임시 컬럼 제거
        df = df.drop(['first_lon', 'first_lat'], axis=1)
        
        print(f"V2AIX 좌표 변환 완료 - 범위: X({df['pos_x'].min():.1f}~{df['pos_x'].max():.1f}m), Y({df['pos_y'].min():.1f}~{df['pos_y'].max():.1f}m)")
        
        return df
    
    # ---------- VeReMi 데이터 로딩 ----------
    def extract_cam_features_veremi(self, json_data: List[Dict], ground_truth: List[Dict], scenario_id: str = "") -> List[Dict]:
        """VeReMi 데이터에서 CAM 메시지 특성 추출 (GroundTruth와 매핑)"""
        features = []
        attacker_info = {}
        
        # Ground Truth에서 공격자 정보 추출
        for gt_entry in ground_truth:
            if gt_entry.get('attackerType', 0) > 0:
                key = (gt_entry['time'], gt_entry['sender'], gt_entry['messageID'])
                attacker_info[key] = gt_entry['attackerType']

        # JSON 로그에서 특성 추출
        for entry in json_data:
            if entry.get('type') == 3:  # CAM 메시지 타입
                try:
                    key = (entry['sendTime'], entry['sender'], entry['messageID'])
                    is_attacker = key in attacker_info
                    attacker_type = attacker_info.get(key, 0)
                    
                    # VeReMi 원본 데이터 (시뮬레이션 단위)
                    pos_x_sim = entry['pos'][0]  # 미터 (시뮬레이션 좌표)
                    pos_y_sim = entry['pos'][1]  # 미터 (시뮬레이션 좌표) 
                    pos_z_sim = entry['pos'][2]  # 미터
                    spd_x = entry['spd'][0]      # m/s
                    spd_y = entry['spd'][1]      # m/s
                    spd_z = entry['spd'][2]      # m/s
                    
                    # 속도와 헤딩 계산
                    speed = float(np.sqrt(spd_x**2 + spd_y**2))  # m/s
                    heading = np.degrees(np.arctan2(spd_y, spd_x))  # degree
                    heading = (heading + 360) % 360  # 0~360도 범위로 조정
                    # VeReMi는 미터 단위 시뮬레이션 좌표를 그대로 사용
                    pos_x = pos_x_sim  # 미터 (시뮬레이션 좌표)
                    pos_y = pos_y_sim  # 미터 (시뮬레이션 좌표)
                    pos_z = pos_z_sim  # 미터
                    
                    feature = {
                        'timestamp': entry['sendTime'],  # s (ETSI CAM과 동일)
                        'station_id': entry['sender'],
                        'pos_x': pos_x,
                        'pos_y': pos_y,
                        'pos_z': pos_z,
                        'spd_x': spd_x,  # m/s (ETSI CAM과 동일)
                        'spd_y': spd_y,  # m/s (ETSI CAM과 동일)
                        'spd_z': spd_z,  # m/s (ETSI CAM과 동일)
                        'heading': heading,  # degree (ETSI CAM과 동일)
                        'speed': speed,      # m/s (ETSI CAM과 동일)
                        'acceleration': 0.0,  # 이후 동적 피처 계산에서 재계산
                        'curvature': 0.0,     # 이후 동적 피처 계산에서 재계산
                        'is_attacker': is_attacker,
                        'attacker_type': attacker_type,
                        'scenario_id': scenario_id
                    }
                    features.append(feature)
                except (KeyError, TypeError) as e:
                    continue
                    
        return features

    def load_veremi_data(self, json_log_path: str, ground_truth_path: str, 
                        target_attacker_types: Optional[List[int]] = None) -> pd.DataFrame:
        """VeReMi 데이터 로딩 (특정 공격 타입 필터링 지원)"""
        try:
            # Ground Truth 파일 읽기
            ground_truth = []
            with open(ground_truth_path, 'r') as f:
                for line in f:
                    ground_truth.append(json.loads(line.strip()))
            
            # JSON 로그 파일 읽기
            json_data = []
            with open(json_log_path, 'r') as f:
                for line in f:
                    json_data.append(json.loads(line.strip()))
            
            # JSON 로그 파일 경로에서 시나리오 ID 추출
            scenario_path = Path(json_log_path)
            current_path = scenario_path.parent  # results
            
            # results 디렉토리에서 최상위 시나리오 디렉토리까지 거슬러 올라가기
            while current_path.name not in ['all', 'VeReMi_Data'] and current_path.parent != current_path:
                if current_path.name.startswith('veins_'):  # 시나리오 디렉토리 패턴
                    scenario_id = current_path.name
                    break
                current_path = current_path.parent
            else:
                scenario_id = scenario_path.parent.name  # 기본값
                
            # CAM 특성 추출
            features = self.extract_cam_features_veremi(json_data, ground_truth, scenario_id)
            df = pd.DataFrame(features)
            
            # 특정 공격 타입 필터링
            if not df.empty and target_attacker_types is not None:
                mask = df['attacker_type'].isin(target_attacker_types)
                df = df[mask].reset_index(drop=True)
                print(f"  공격 타입 {target_attacker_types}로 필터링: {len(df)} 레코드")
            
            if not df.empty:
                df['dataset'] = 'veremi'

                
                
            return df
            
        except Exception as e:
            print(f"VeReMi 데이터 로딩 실패: {json_log_path} - {str(e)}")
            return pd.DataFrame()

    # ---------- VeReMi 데이터셋 유틸리티 ----------
    def iter_veremi_pairs(self, roots, target_attacker_types=None):
        """VeReMi 데이터셋에서 (JSONlog, GroundTruth) 파일 쌍을 찾는 제너레이터"""
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
                
                # 같은 디렉토리의 로그들
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

    def load_veremi_dataset(self, veremi_root: str, max_files: int = 1000, 
                           directory_filter_types: Optional[List[int]] = None,
                           data_filter_types: Optional[List[int]] = None,
                           max_scenarios: Optional[int] = None) -> pd.DataFrame:
        """VeReMi 데이터셋 전체 로딩 (여러 파일 처리, 시나리오 디렉터리 제한 지원)"""
        veremi_dfs = []
            
        print(f"VeReMi 로그 로딩 중 (최대 {max_files} 파일, 최대 {max_scenarios if max_scenarios else '제한 없음'} 시나리오)...")
        
        if directory_filter_types:
            print(f"디렉토리 필터링: 공격 타입 {directory_filter_types}")
        
        all_pairs = list(self.iter_veremi_pairs(veremi_root, target_attacker_types=directory_filter_types))
        print(f"필터링된 디렉토리에서 {len(all_pairs)}개 파일 쌍 발견")
        
        # 시나리오별로 그룹핑
        scenario_to_pairs = {}
        for log_path, gt_path in all_pairs:
            # 시나리오 디렉토리명 추출 (veins_* 또는 상위 디렉토리)
            scenario_dir = Path(log_path).parent
            while not scenario_dir.name.startswith('veins_') and scenario_dir.parent != scenario_dir:
                scenario_dir = scenario_dir.parent
            scenario_id = scenario_dir.name
            scenario_to_pairs.setdefault(scenario_id, []).append((log_path, gt_path))
        
        # 시나리오 제한 적용
        scenario_ids = list(scenario_to_pairs.keys())
        if max_scenarios is not None:
            scenario_ids = scenario_ids[:max_scenarios]
            print(f"시나리오 제한 적용: {len(scenario_ids)}개 시나리오만 사용")
        
        total_files = 0
        for scenario_id in scenario_ids:
            pairs = scenario_to_pairs[scenario_id]
            for log_path, gt_path in pairs:
                if total_files >= max_files:
                    break
                file_name = log_path.split('/')[-1] if '/' in log_path else log_path.split('\\')[-1]
                if (total_files + 1) % 50 == 0:
                    print(f"처리 중: {total_files+1}/{min(len(all_pairs), max_files)} - {file_name}")
                df = self.load_veremi_data(log_path, gt_path, data_filter_types)
                if not df.empty:
                    veremi_dfs.append(df)
                total_files += 1
            if total_files >= max_files:
                break
        
        # 모든 VeReMi 데이터 통합
        veremi_df = pd.concat(veremi_dfs, ignore_index=True) if veremi_dfs else pd.DataFrame()

        # VeReMi 중복 메시지 제거 (전체 데이터에서)
        if not veremi_df.empty:
            print(f"중복 제거 전: {len(veremi_df)} 레코드")
            # sendTime, sender, scenario_id 조합으로 중복 제거
            veremi_df = veremi_df.drop_duplicates(
                subset=['timestamp', 'station_id', 'scenario_id'], 
                keep='first'
            ).reset_index(drop=True)
            print(f"중복 제거 후: {len(veremi_df)} 레코드")

            # 정렬 추가
            veremi_df = veremi_df.sort_values(
                ['scenario_id', 'station_id', 'timestamp']
            ).reset_index(drop=True)

            # VeReMi 공격 타입별 통계 출력
            print("=" * 80)
            print(f"VeReMi 데이터 로딩 완료: {len(veremi_df)} 레코드")
            attacker_stats = veremi_df['attacker_type'].value_counts().sort_index()
            print("공격 타입별 분포:")
            for att_type, count in attacker_stats.items():
                att_name = "정상" if att_type == 0 else f"공격 타입 {att_type}"
                print(f"  {att_name}: {count} 레코드 ({count/len(veremi_df)*100:.1f}%)")

        return veremi_df

    def analyze_data_quality(self, df: pd.DataFrame, dataset_name: str = "") -> dict:
        """데이터 품질 분석"""
        quality_report = {
            'dataset_name': dataset_name,
            'rows': len(df),
            'columns': len(df.columns),
            'missing_total': df.isnull().sum().sum(),
            'duplicates': df.duplicated().sum(),
            'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
            'object_columns': len(df.select_dtypes(include=['object']).columns)
        }
        
        # is_attacker 컬럼 분석
        if 'is_attacker' in df.columns:
            attack_count = df['is_attacker'].sum()
            quality_report['attack_count'] = int(attack_count)
            quality_report['attack_ratio'] = float(attack_count / len(df)) if len(df) > 0 else 0.0
            quality_report['normal_count'] = int(len(df) - attack_count)
        
        print(f"📊 {dataset_name} 데이터 품질 분석:")
        print(f"  - 행 수: {quality_report['rows']:,}")
        print(f"  - 컬럼 수: {quality_report['columns']}")
        print(f"  - 결측값: {quality_report['missing_total']:,}")
        print(f"  - 중복: {quality_report['duplicates']:,}")
        
        if 'attack_count' in quality_report:
            print(f"  - 정상: {quality_report['normal_count']:,} ({1-quality_report['attack_ratio']:.1%})")
            print(f"  - 공격: {quality_report['attack_count']:,} ({quality_report['attack_ratio']:.1%})")
        
        return quality_report

    def get_unit_info(self) -> dict:  
        """현재 설정된 단위 정보 반환"""
        return {
            'standardize_units': self.standardize_units,
            'v2aix_units': {
                'position': 'meter (local coordinates)',
                'speed': 'm/s',
                'heading': 'degree',
                'timestamp': 'second (converted from nanosecond)'
            },
            'veremi_units_original': {
                'position': 'meter (simulation coordinates)',
                'speed': 'm/s',
                'heading': 'degree (calculated from velocity)',
                'timestamp': 'second'
            },
            'veremi_units_final': {
                'position': 'meter (simulation coordinates)',
                'speed': 'm/s',
                'heading': 'degree', 
                'timestamp': 'second'
            },
            'coordinate_system': 'meters (unified)',
            'etsi_cam_standard': self.ETSI_UNITS
        }

    def save_standardized_data(self, v2aix_df=None, veremi_df=None):
        """이미 로딩된 데이터를 z_test 디렉토리에 저장"""
        print("\n" + "=" * 80)
        print("데이터 저장 중...")
        print("=" * 80)
        
        # 데이터가 없으면 빈 DataFrame 생성
        if v2aix_df is None:
            v2aix_df = pd.DataFrame()
        if veremi_df is None:
            veremi_df = pd.DataFrame()

        # z_test 디렉토리 생성 및 저장
        os.makedirs("z_test", exist_ok=True)
        
        # 개별 데이터셋 저장
        if not v2aix_df.empty:
            v2aix_path = "z_test/v2aix_etsi_standardized.csv"
            v2aix_df.to_csv(v2aix_path, index=False)
            print(f"💾 V2AIX 데이터 저장: {v2aix_path}")
        
        if not veremi_df.empty:
            veremi_path = "z_test/veremi_etsi_standardized.csv"
            veremi_df.to_csv(veremi_path, index=False)
            print(f"💾 VeReMi 데이터 저장: {veremi_path}")
        
        # 단위 정보도 저장
        unit_info = self.get_unit_info()
        unit_info_path = "z_test/unit_conversion_info.json"
        with open(unit_info_path, 'w', encoding='utf-8') as f:
            json.dump(unit_info, f, indent=2, ensure_ascii=False)
        print(f"💾 단위 변환 정보 저장: {unit_info_path}")
        
        # 요약 정보 저장
        if not v2aix_df.empty or not veremi_df.empty:
            combined_df = pd.concat([v2aix_df, veremi_df], ignore_index=True)
        else:
            combined_df = pd.DataFrame()
            
        total_attackers = len(combined_df[combined_df['is_attacker'] == 1]) if not combined_df.empty else 0
        total_normal = len(combined_df[combined_df['is_attacker'] == 0]) if not combined_df.empty else 0

        summary = {
            'total_records': len(combined_df),
            'v2aix_records': len(v2aix_df) if not v2aix_df.empty else 0,
            'veremi_records': len(veremi_df) if not veremi_df.empty else 0,
            'normal_records': total_normal if not combined_df.empty else 0,
            'attack_records': total_attackers if not combined_df.empty else 0,
            'attack_ratio': total_attackers/(total_normal+total_attackers) if not combined_df.empty and (total_normal+total_attackers) > 0 else 0,
            'columns': list(combined_df.columns) if not combined_df.empty else [],
            'standardization_applied': True,
            'etsi_cam_compliant': True
        }
        
        summary_path = "z_test/data_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"💾 데이터 요약 정보 저장: {summary_path}")

        print("\n✅ 데이터 저장 완료!")
        print(f"📁 저장 위치: z_test/ 디렉토리")
        return combined_df if not combined_df.empty else None


def main():
    """데이터 로딩 예제"""
    
    # 데이터 로더 초기화
    loader = V2XDataLoader()
    
    # V2AIX 데이터 로딩 예제
    print("=" * 80)
    print("V2AIX 데이터 로딩 예제")
    print("=" * 80)
    
    v2aix_path = "V2AIX_Data/json/Mobile/V2X-only"
    v2aix_df = loader.load_v2aix_data(v2aix_path, max_files=10000)
    
    if not v2aix_df.empty:
        loader.analyze_data_quality(v2aix_df, "V2AIX")
    else:
        print("V2AIX 데이터를 찾을 수 없습니다.")
    
    # VeReMi 데이터 로딩 예제
    print("\n" + "=" * 80)
    print("VeReMi 데이터 로딩 예제")
    print("=" * 80)
    
    veremi_root = "VeReMi_Data/all" 
    
    # 특정 공격 타입만 로딩
    print("특정 공격 타입만 로딩 (타입 1, 2):")
    # directory_filter_types: 디렉토리(시나리오)에서 공격 타입 1, 2만 포함
    # data_filter_types: 실제 데이터에서 정상(0), 공격 타입 1, 2만 포함
    veremi_df_filtered = loader.load_veremi_dataset(
        veremi_root=veremi_root,
        max_files=10000,
        directory_filter_types=[0, 8],   # 시나리오 디렉토리에서 공격 타입 1, 2만
        data_filter_types=[0, 8],     # 데이터에서 정상(0), 공격 1, 2만
        max_scenarios=45               # 시나리오 개수 제한 없음 (필요시 정수로 제한)
    )
    
    if not veremi_df_filtered.empty:
        loader.analyze_data_quality(veremi_df_filtered, "VeReMi (필터링)")
    
    # 통합 데이터 예제
    print("\n" + "=" * 80)
    print("V2AIX + VeReMi 통합 데이터")
    print("=" * 80)
    
    if not v2aix_df.empty and not veremi_df_filtered.empty:
        combined_df = pd.concat([v2aix_df, veremi_df_filtered], ignore_index=True)
        print(f"통합 데이터: {len(combined_df)} 레코드")
        
        # 전체 데이터셋의 공격/정상 비율
        total_attackers = len(combined_df[combined_df['is_attacker'] == 1])
        total_normal = len(combined_df[combined_df['is_attacker'] == 0])
        print(f"정상: {total_normal}, 공격: {total_attackers}")
        if total_attackers > 0:
            print(f"공격 비율: {total_attackers/(total_normal+total_attackers)*100:.2f}%")
    
    print("\n" + "=" * 80)
    print("데이터 로딩 완료!")
    print("=" * 80)
    
    # 로딩된 데이터를 저장
    loader.save_standardized_data(v2aix_df, veremi_df_filtered)


if __name__ == "__main__":
    main()