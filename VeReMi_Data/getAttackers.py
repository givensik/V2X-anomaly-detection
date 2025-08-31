import json
import glob
import os
import re
from collections import defaultdict

def extract_attackers_by_directory():
    """
    VeReMi_Data/all/ 안의 각 디렉토리별로 공격자 sender를 찾아서,
    그 sender가 보낸 모든 메시지를 중복 없이 저장
    """
    base_dir = "all/"
    output_dir = "VeReMi_Data/attackermsg"
    os.makedirs(output_dir, exist_ok=True)

    try:
        scenario_dirs = []
        for item in os.listdir(base_dir):
            item_path = os.path.join(base_dir, item)
            if os.path.isdir(item_path):
                results_path = os.path.join(item_path, "veins-maat", "simulations", "securecomm2018", "results")
                if os.path.exists(results_path):
                    scenario_dirs.append((item, results_path))

        print(f"총 {len(scenario_dirs)}개의 시나리오 디렉토리를 찾았습니다.")

        for scenario_name, results_path in scenario_dirs:
            print(f"\n[시나리오: {scenario_name}]")

            # 1. GroundTruth에서 공격자 sender 추출
            ground_truth_file = os.path.join(results_path, "GroundTruthJSONlog.json")
            if not os.path.exists(ground_truth_file):
                print(f"  GroundTruthJSONlog.json 파일이 없습니다.")
                continue

            attacker_senders = set()
            with open(ground_truth_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            data = json.loads(line)
                            if data.get('attackerType', 0) > 0:
                                sender = data.get('senderID') or data.get('sender') or data.get('nodeID')
                                if sender:
                                    attacker_senders.add(sender)
                        except Exception:
                            continue
            print(f"  공격자 sender {len(attacker_senders)}명: {sorted(attacker_senders)}")

            # 2. JSONlog에서 해당 sender가 보낸 모든 메시지 추출 (중복 제거)
            jsonlog_pattern = os.path.join(results_path, "JSONlog-*.json")
            jsonlog_files = glob.glob(jsonlog_pattern)
            seen_msg_ids = set()
            sender_msgs = []

            for jsonlog_file in sorted(jsonlog_files):
                with open(jsonlog_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                data = json.loads(line)
                                sender = data.get('senderID') or data.get('sender') or data.get('nodeID')
                                msg_id = data.get('messageID')
                                if sender in attacker_senders and msg_id not in seen_msg_ids:
                                    data['scenario'] = scenario_name
                                    data['source_file'] = os.path.basename(jsonlog_file)
                                    sender_msgs.append(data)
                                    seen_msg_ids.add(msg_id)
                            except Exception:
                                continue

            print(f"  추출된 메시지 {len(sender_msgs)}개 (중복 제거됨)")

            # 3. 저장
            if sender_msgs:
                scenario_output_file = os.path.join(output_dir, f"attacker_senders_msgs_{scenario_name}.json")
                with open(scenario_output_file, 'w', encoding='utf-8') as f:
                    for data in sender_msgs:
                        f.write(json.dumps(data, ensure_ascii=False) + '\n')
                print(f"  저장됨: {scenario_output_file}")

    except Exception as e:
        print(f"오류 발생: {e}")

if __name__ == "__main__":
    extract_attackers_by_directory()