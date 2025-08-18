# veremi_preprocess_min_t3.py
import json, glob, argparse
from pathlib import Path
import pandas as pd
import numpy as np

# ---------- JSONL reader ----------
def read_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                # 깨진 라인이 섞여 있어도 전체 처리는 계속
                continue

# ---------- 안전 접근 & 표준화 ----------
def _first(*vals):
    for v in vals:
        if v is not None:
            return v
    return None

def normalize_msg_id(v):
    # messageID가 str/float/int로 뒤섞일 수 있어 통일(가능하면 int, 아니면 str)
    if v is None:
        return None
    try:
        # "123" -> 123, 123.0 -> 123
        iv = int(float(v))
        return iv
    except Exception:
        return str(v)

def normalize_sender(v):
    # sender도 문자열/숫자 혼재 가능 → 가능하면 int, 아니면 str
    if v is None:
        return None
    try:
        return int(str(v))
    except Exception:
        return str(v)

# ---------- loaders (type=3 only + GT) ----------
def load_type3(glob_pat: str) -> pd.DataFrame:
    rows = []
    for p in glob.glob(glob_pat):
        for obj in read_jsonl(p):
            if obj.get("type") != 3:
                continue
            msg_id = _first(obj.get("messageID"), obj.get("messageId"))
            rows.append(dict(
                rcvTime=pd.to_numeric(obj.get("rcvTime"), errors="coerce"),
                sendTime=pd.to_numeric(obj.get("sendTime"), errors="coerce"),
                sender=normalize_sender(obj.get("sender")),
                receiver=_first(obj.get("receiver"), obj.get("rx")),
                messageID=normalize_msg_id(msg_id),
                RSSI=pd.to_numeric(obj.get("RSSI"), errors="coerce"),
                _src=Path(p).name,
            ))
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # 필수 필드 정리
    df = df.dropna(subset=["sender", "messageID", "rcvTime"])
    # 중복 수신 기록은 (sender, messageID, receiver, rcvTime) 기준으로 보존(수신자별 레코드 유지)
    df = df.drop_duplicates(subset=["sender", "messageID", "receiver", "rcvTime"])
    # 정렬
    df = df.sort_values(["sender", "rcvTime"]).reset_index(drop=True)
    return df

def load_gt(path: str) -> pd.DataFrame:
    if not path or not Path(path).exists():
        return pd.DataFrame()
    rows = []
    for obj in read_jsonl(path):
        # 일부 배포본은 type을 안 쓰거나 4가 아닌 경우도 있어 관대하게 수용
        msg_id = _first(obj.get("messageID"), obj.get("messageId"))
        atk = _first(obj.get("attackerType"), obj.get("attackType"))
        rows.append(dict(
            time=pd.to_numeric(obj.get("time"), errors="coerce"),
            sender=normalize_sender(obj.get("sender")),
            messageID=normalize_msg_id(msg_id),
            attackerType=pd.to_numeric(atk, errors="coerce")
        ))
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # 필요 키 유지 + 중복 제거
    df = df.dropna(subset=["sender", "messageID"])
    df = df.drop_duplicates(subset=["sender", "messageID"], keep="last")
    # 바이너리 라벨: 0(정상) 외에는 1
    df["label"] = df["attackerType"].apply(lambda a: 0 if (pd.notna(a) and float(a) == 0.0) else (1 if pd.notna(a) else np.nan))
    return df

# ---------- preprocess (type=3 + GT) ----------
def preprocess_veremi_t3_only(type3_glob: str, gt_file: str|None, out_dir: str):
    outp = Path(out_dir); outp.mkdir(parents=True, exist_ok=True)

    df3 = load_type3(type3_glob)
    if df3.empty:
        print(f"[warn] no type=3 messages from {type3_glob}")
        return

    dfg = load_gt(gt_file) if gt_file else pd.DataFrame()

    # (sender, messageID)로 GT 결합 (정석)
    if not dfg.empty:
        df = df3.merge(dfg[["sender","messageID","attackerType","label"]],
                       on=["sender","messageID"], how="left")
    else:
        df = df3.copy()
        df["attackerType"] = np.nan
        df["label"] = np.nan

    # 컬럼 정리
    keep = [
        "rcvTime","sendTime","sender","receiver","messageID","RSSI",
        "attackerType","label","_src"
    ]
    for c in keep:
        if c not in df.columns:
            df[c] = np.nan
    df = df[keep].sort_values(["sender","rcvTime"]).reset_index(drop=True)

    # 저장
    df.to_parquet(outp/"veremi_t3_minimal.parquet", index=False)
    df.to_csv(outp/"veremi_t3_minimal.csv", index=False, na_rep="NaN")
    print(f"[OK] Saved: {outp/'veremi_t3_minimal.parquet'} ({len(df):,} rows), {outp/'veremi_t3_minimal.csv'}")

# ---------- CLI ----------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="VeReMi preprocessing (type=3 only, minimal features)")
    ap.add_argument("--type3_glob", default="results/JSONlog-*.json", help="Path pattern for type=3 logs (JSON Lines)")
    ap.add_argument("--gt_file", default="results/GroundTruthJSONlog.json", help="Ground truth JSON Lines file (optional)")
    ap.add_argument("--out_dir", default="out", help="Output directory")
    args = ap.parse_args()

    preprocess_veremi_t3_only(args.type3_glob, args.gt_file, args.out_dir)
