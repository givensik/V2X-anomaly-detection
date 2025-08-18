import json
from pathlib import Path

# 경로 설정
data_dir = Path("results")
groundtruth_file = data_dir / "GroundTruthJSONlog.json"
sample_log_file = next(f for f in data_dir.glob("JSONlog-*.json"))

# JSON Lines 형식 로드 함수
def load_json_lines(file_path):
    with open(file_path, "r") as f:
        return [json.loads(line) for line in f if line.strip()]

# 데이터 로드
groundtruth_data = load_json_lines(groundtruth_file)
sample_log_data = load_json_lines(sample_log_file)

print("GroundTruth keys:", list(groundtruth_data[0].keys()))
print("Message Log keys:", list(sample_log_data[0].keys()))

# 키 비교
def compare_keys(list1, list2):
    keys1 = set(list1[0].keys())
    keys2 = set(list2[0].keys())
    return {
        "only_in_groundtruth": sorted(keys1 - keys2),
        "only_in_message_log": sorted(keys2 - keys1),
        "common_keys": sorted(keys1 & keys2)
    }

result = compare_keys(groundtruth_data, sample_log_data)

print("\n🔍 Key differences:")
print(json.dumps(result, indent=2))
