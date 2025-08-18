import json, math, glob, os
from pathlib import Path
import pandas as pd
from typing import Any, Dict, Optional

# ---------- helpers ----------
def safe_get(d: Dict[str, Any], path: str, default=None):
    cur = d
    try:
        for p in path.split('.'):
            if p.endswith(']'):
                # list index
                name, idx = p[:-1].split('[')
                cur = cur.get(name, [])
                cur = cur[int(idx)]
            else:
                cur = cur.get(p, {})
        # leaf reached
        return cur if cur != {} else default
    except Exception:
        return default

def haversine_m(lat1, lon1, lat2, lon2):
    if any(pd.isna(x) for x in (lat1, lon1, lat2, lon2)):
        return float('nan')
    R = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi  = math.radians(lat2 - lat1)
    dlamb = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlamb/2)**2
    return 2 * R * math.asin(math.sqrt(a))

def bearing_deg(lat1, lon1, lat2, lon2):
    if any(pd.isna(x) for x in (lat1, lon1, lat2, lon2)):
        return float('nan')
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dlamb = math.radians(lon2 - lon1)
    y = math.sin(dlamb) * math.cos(phi2)
    x = math.cos(phi1)*math.sin(phi2) - math.sin(phi1)*math.cos(phi2)*math.cos(dlamb)
    brng = math.degrees(math.atan2(y, x))
    return (brng + 360.0) % 360.0

def angdiff_deg(a, b):
    # smallest signed difference (a - b)
    d = (a - b + 180) % 360 - 180
    return d

# ---------- parser for one JSON file ----------
def parse_v2aix_cam_file(path: str) -> pd.DataFrame:
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    rows = []
    for rec in data.get('/v2x/cam', []):
        ts_ns = rec.get('recording_timestamp_nsec')
        ts_sec = (ts_ns / 1e9) if isinstance(ts_ns, (int, float)) else None  # ★ 초 단위로 변환

        msg = rec.get('message', {})
        station_id = safe_get(msg, 'header.station_id.value')

        # containers
        lat_raw = safe_get(msg, 'cam.cam_parameters.basic_container.reference_position.latitude.value')
        lon_raw = safe_get(msg, 'cam.cam_parameters.basic_container.reference_position.longitude.value')
        alt_raw = safe_get(msg, 'cam.cam_parameters.basic_container.reference_position.altitude.altitude_value.value')

        speed_raw   = safe_get(msg, 'cam.cam_parameters.high_frequency_container.basic_vehicle_container_high_frequency.speed.speed_value.value')
        heading_raw = safe_get(msg, 'cam.cam_parameters.high_frequency_container.basic_vehicle_container_high_frequency.heading.heading_value.value')
        ax_long_raw = safe_get(msg, 'cam.cam_parameters.high_frequency_container.basic_vehicle_container_high_frequency.longitudinal_acceleration.longitudinal_acceleration_value.value')
        ax_lat_raw  = safe_get(msg, 'cam.cam_parameters.high_frequency_container.basic_vehicle_container_high_frequency.lateral_acceleration.lateral_acceleration_value.value')
        yaw_raw     = safe_get(msg, 'cam.cam_parameters.high_frequency_container.basic_vehicle_container_high_frequency.yaw_rate.yaw_rate_value.value')

        # unit conversions
        lat_deg = lat_raw / 1e7 if isinstance(lat_raw, (int, float)) else None
        lon_deg = lon_raw / 1e7 if isinstance(lon_raw, (int, float)) else None
        heading_deg = heading_raw / 10.0 if isinstance(heading_raw, (int, float)) else None
        speed_ms    = speed_raw / 100.0 if isinstance(speed_raw, (int, float)) else None
        ax_long_ms2 = ax_long_raw / 10.0 if isinstance(ax_long_raw, (int, float)) else None
        ax_lat_ms2  = ax_lat_raw / 10.0 if isinstance(ax_lat_raw, (int, float)) else None
        yaw_dps     = yaw_raw / 100.0 if isinstance(yaw_raw, (int, float)) else None

        # quick sanity (옵션)
        if isinstance(speed_ms, (int, float)) and (speed_ms < 0 or speed_ms > 80):
            speed_ms = None  # 비현실 범위 제거

        rows.append(dict(
            file=Path(path).name,
            ts=ts_sec,                     # ★ 초 단위
            station_id=station_id,
            lat_deg=lat_deg, lon_deg=lon_deg,
            speed_ms=speed_ms, heading_deg=heading_deg,
            ax_long_ms2=ax_long_ms2, ax_lat_ms2=ax_lat_ms2, yaw_rate_dps=yaw_dps,
            alt_raw=alt_raw
        ))

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # 기본 정렬 & 중복 제거
    df = df.dropna(subset=['ts', 'station_id']).sort_values(['station_id','ts']).reset_index(drop=True)
    df = df.drop_duplicates(subset=['station_id','ts'], keep='first')

    # per-station dynamics
    out = []
    for sid, g in df.groupby('station_id', sort=False):
        g = g.sort_values('ts').copy()
        g['dt'] = g['ts'].diff()  # ★ 단위: 초(sec)

        g['lat_prev'] = g['lat_deg'].shift(1)
        g['lon_prev'] = g['lon_deg'].shift(1)
        g['dist_m'] = g.apply(lambda r: haversine_m(r['lat_prev'], r['lon_prev'], r['lat_deg'], r['lon_deg']), axis=1)

        # 너무 작은 이동은 불안정 → NaN
        small_move = g['dist_m'] < 0.1
        g.loc[small_move, 'dist_m'] = float('nan')

        # path bearing & 변화
        g['bearing_deg_path'] = g.apply(lambda r: bearing_deg(r['lat_prev'], r['lon_prev'], r['lat_deg'], r['lon_deg']), axis=1)
        g['bearing_change_deg'] = angdiff_deg(g['bearing_deg_path'], g['bearing_deg_path'].shift(1))

        # curvature (rad/m)
        g['curvature_1pm'] = (g['bearing_change_deg'].abs() * math.pi/180.0) / g['dist_m']

        # position-derived dynamics (sec 기반)
        g['speed_from_pos_ms'] = g['dist_m'] / g['dt']
        g['acc_ms2']  = g['speed_ms'].diff() / g['dt']
        g['jerk_ms3'] = g['acc_ms2'].diff() / g['dt']

        # heading vs path heading
        g['heading_consistency'] = g.apply(
            lambda r: angdiff_deg(r['heading_deg'], r['bearing_deg_path'])
            if pd.notna(r.get('heading_deg')) and pd.notna(r.get('bearing_deg_path')) else float('nan'),
            axis=1
        )

        # 비정상 dt(<=0) 제거
        g = g[(g['dt'].isna()) | (g['dt'] > 0)]

        out.append(g.drop(columns=['lat_prev','lon_prev']))
    res = pd.concat(out, ignore_index=True)
    return res


def run(input_glob: str, out_parquet: str, out_csv: Optional[str] = None):
    # __file__이 없는 환경(노트북) 대비
    base_dir = Path(__file__).parent if '__file__' in globals() else Path.cwd()
    files = sorted((base_dir / input_glob).glob("*")) if any(ch in input_glob for ch in ['*','?','[']) else [base_dir / input_glob]

    # glob 패턴일 때 재귀 확장
    if any(ch in input_glob for ch in ['*','?','[']):
        files = sorted(base_dir.glob(input_glob))

    if not files:
        print(f"[경고] 입력 파일이 없습니다: {input_glob} (base={base_dir})")
        return

    all_df = []
    for p in files:
        if p.is_file():
            df = parse_v2aix_cam_file(str(p))
            if not df.empty:
                all_df.append(df)

    if not all_df:
        print("No CAM records found.")
        return

    full = pd.concat(all_df, ignore_index=True)

    # station_id, ts 기준 중복/역순 방지 최종 클린업
    full = full.drop_duplicates(subset=['station_id','ts'], keep='first')
    full = full.sort_values(['station_id','ts'])
    full = full[(full['dt'].isna()) | (full['dt'] > 0)]
    full['is_track_start'] = full.groupby('station_id').cumcount() == 0
    
    Path(Path(out_parquet).parent).mkdir(parents=True, exist_ok=True)
    full.to_parquet(out_parquet, index=False)
    if out_csv:
        # full.to_csv(out_csv, index=False)
        full.to_csv(out_csv, index=False, na_rep="NaN") # 
        
        
    print(f"Saved: {out_parquet} ({len(full):,} rows)" + (f", {out_csv}" if out_csv else ""))


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="V2AIX CAM JSON to Parquet converter")
    parser.add_argument("input_glob", type=str, help="Input file glob pattern (e.g., 'json/Mobile/V2X-only/Aachen/scenarios/*.json')")
    parser.add_argument("out_parquet", type=str, help="Output Parquet file path")
    parser.add_argument("--out_csv", type=str, default=None, help="Optional output CSV file path")
    print("V2AIX CAM JSON to Parquet converter")
    args = parser.parse_args()
    run(args.input_glob, args.out_parquet, args.out_csv)