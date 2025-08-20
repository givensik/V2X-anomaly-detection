import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
from typing import Dict, List, Tuple
import os

class VeReMiGroundTruthAnalyzer:
    """VeReMi GroundTruth 파일 분석 클래스"""
    
    def __init__(self, ground_truth_path: str):
        """
        Args:
            ground_truth_path: GroundTruthJSONlog.json 파일 경로
        """
        self.ground_truth_path = ground_truth_path
        self.ground_truth_data = []
        self.attacker_info = {}
        self.load_ground_truth()
    
    def load_ground_truth(self):
        """GroundTruth 파일 로드"""
        print(f"Loading GroundTruth from: {self.ground_truth_path}")
        
        with open(self.ground_truth_path, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    self.ground_truth_data.append(entry)
                except json.JSONDecodeError as e:
                    print(f"Error parsing line: {e}")
                    continue
        
        print(f"Loaded {len(self.ground_truth_data)} GroundTruth entries")
    
    def analyze_attackers(self) -> Dict:
        """공격자 정보 분석"""
        print("\n=== ATTACKER ANALYSIS ===")
        
        # 공격자 타입별 통계
        attacker_types = [entry['attackerType'] for entry in self.ground_truth_data if entry['attackerType'] > 0]
        attacker_counter = Counter(attacker_types)
        
        print(f"Total messages: {len(self.ground_truth_data)}")
        print(f"Attack messages: {len(attacker_types)}")
        print(f"Normal messages: {len(self.ground_truth_data) - len(attacker_types)}")
        print(f"Attack ratio: {len(attacker_types) / len(self.ground_truth_data) * 100:.2f}%")
        
        print("\nAttacker types distribution:")
        for attacker_type, count in attacker_counter.most_common():
            print(f"  Type {attacker_type}: {count} messages ({count/len(attacker_types)*100:.1f}%)")
        
        # 공격자별 통계
        attacker_senders = [entry['sender'] for entry in self.ground_truth_data if entry['attackerType'] > 0]
        sender_counter = Counter(attacker_senders)
        
        print(f"\nUnique attackers: {len(sender_counter)}")
        print("Top attackers by message count:")
        for sender, count in sender_counter.most_common(10):
            print(f"  Sender {sender}: {count} attack messages")
        
        return {
            'attacker_types': attacker_counter,
            'attacker_senders': sender_counter,
            'total_messages': len(self.ground_truth_data),
            'attack_messages': len(attacker_types),
            'normal_messages': len(self.ground_truth_data) - len(attacker_types)
        }
    
    def create_attacker_mapping(self) -> Dict:
        """공격자 매핑 딕셔너리 생성"""
        print("\nCreating attacker mapping...")
        
        self.attacker_info = {} # 매핑 초기화
        for entry in self.ground_truth_data:
            if entry['attackerType'] > 0:  # 공격자
                # 튜플 키를 문자열 키로 변경
                key = f"{entry['time']}_{entry['sender']}_{entry['messageID']}"
                self.attacker_info[key] = {
                    'attackerType': entry['attackerType'],
                    'time': entry['time'],
                    'sender': entry['sender'],
                    'messageID': entry['messageID']
                }
        
        print(f"Created mapping for {len(self.attacker_info)} attack messages")
        return self.attacker_info
    
    def analyze_temporal_patterns(self) -> pd.DataFrame:
        """시간적 패턴 분석"""
        print("\n=== TEMPORAL PATTERN ANALYSIS ===")
        
        # 시간별 공격 분포
        attack_times = [entry['time'] for entry in self.ground_truth_data if entry['attackerType'] > 0]
        normal_times = [entry['time'] for entry in self.ground_truth_data if entry['attackerType'] == 0]
        
        # 시간 구간별 분석
        time_bins = np.linspace(min(attack_times + normal_times), 
                              max(attack_times + normal_times), 20)
        
        attack_hist, _ = np.histogram(attack_times, bins=time_bins)
        normal_hist, _ = np.histogram(normal_times, bins=time_bins)
        
        # 데이터프레임 생성
        df_temporal = pd.DataFrame({
            'time_bin_start': time_bins[:-1],
            'time_bin_end': time_bins[1:],
            'attack_count': attack_hist,
            'normal_count': normal_hist,
            'total_count': attack_hist + normal_hist,
            'attack_ratio': attack_hist / (attack_hist + normal_hist + 1e-10) * 100
        })
        
        print("Temporal attack distribution:")
        print(df_temporal[['time_bin_start', 'attack_count', 'normal_count', 'attack_ratio']].head(10))
        
        return df_temporal
    
    def analyze_spatial_patterns(self) -> pd.DataFrame:
        """공간적 패턴 분석"""
        print("\n=== SPATIAL PATTERN ANALYSIS ===")
        
        # 위치 정보 추출
        spatial_data = []
        for entry in self.ground_truth_data:
            if 'pos' in entry:
                spatial_data.append({
                    'time': entry['time'],
                    'sender': entry['sender'],
                    'pos_x': entry['pos'][0],
                    'pos_y': entry['pos'][1],
                    'pos_z': entry['pos'][2],
                    'is_attacker': entry['attackerType'] > 0,
                    'attacker_type': entry['attackerType']
                })
        
        df_spatial = pd.DataFrame(spatial_data)
        
        if not df_spatial.empty:
            print(f"Spatial data shape: {df_spatial.shape}")
            print(f"Attack positions: {len(df_spatial[df_spatial['is_attacker']])}")
            print(f"Normal positions: {len(df_spatial[~df_spatial['is_attacker']])}")
            
            # 위치별 공격 분포
            print("\nPosition ranges:")
            print(f"X: {df_spatial['pos_x'].min():.2f} ~ {df_spatial['pos_x'].max():.2f}")
            print(f"Y: {df_spatial['pos_y'].min():.2f} ~ {df_spatial['pos_y'].max():.2f}")
            print(f"Z: {df_spatial['pos_z'].min():.2f} ~ {df_spatial['pos_z'].max():.2f}")
        
        return df_spatial
    
    def visualize_analysis(self, output_dir: str = "veremi_analysis"):
        """분석 결과 시각화"""
        print(f"\nGenerating visualizations in: {output_dir}")
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 1. 공격자 타입 분포
        attacker_analysis = self.analyze_attackers()
        
        plt.figure(figsize=(15, 10))
        
        # 공격자 타입 분포
        plt.subplot(2, 3, 1)
        attacker_types = list(attacker_analysis['attacker_types'].keys())
        attacker_counts = list(attacker_analysis['attacker_types'].values())
        plt.pie(attacker_counts, labels=[f'Type {t}' for t in attacker_types], autopct='%1.1f%%')
        plt.title('Attacker Type Distribution')
        
        # 메시지 타입 분포
        plt.subplot(2, 3, 2)
        message_types = ['Normal', 'Attack']
        message_counts = [attacker_analysis['normal_messages'], attacker_analysis['attack_messages']]
        plt.bar(message_types, message_counts, color=['blue', 'red'])
        plt.title('Message Type Distribution')
        plt.ylabel('Count')
        
        # 상위 공격자
        plt.subplot(2, 3, 3)
        top_attackers = attacker_analysis['attacker_senders'].most_common(10)
        senders = [f'Sender {s}' for s, _ in top_attackers]
        counts = [c for _, c in top_attackers]
        plt.barh(senders, counts, color='red')
        plt.title('Top Attackers by Message Count')
        plt.xlabel('Attack Message Count')
        
        # 2. 시간적 패턴
        df_temporal = self.analyze_temporal_patterns()
        
        plt.subplot(2, 3, 4)
        plt.plot(df_temporal['time_bin_start'], df_temporal['attack_count'], 'r-', label='Attacks')
        plt.plot(df_temporal['time_bin_start'], df_temporal['normal_count'], 'b-', label='Normal')
        plt.title('Temporal Message Distribution')
        plt.xlabel('Time')
        plt.ylabel('Message Count')
        plt.legend()
        
        plt.subplot(2, 3, 5)
        plt.plot(df_temporal['time_bin_start'], df_temporal['attack_ratio'], 'g-')
        plt.title('Attack Ratio Over Time')
        plt.xlabel('Time')
        plt.ylabel('Attack Ratio (%)')
        
        # 3. 공간적 패턴
        df_spatial = self.analyze_spatial_patterns()
        
        if not df_spatial.empty:
            plt.subplot(2, 3, 6)
            normal_pos = df_spatial[~df_spatial['is_attacker']]
            attack_pos = df_spatial[df_spatial['is_attacker']]
            
            plt.scatter(normal_pos['pos_x'], normal_pos['pos_y'], 
                       c='blue', alpha=0.6, s=10, label='Normal')
            plt.scatter(attack_pos['pos_x'], attack_pos['pos_y'], 
                       c='red', alpha=0.8, s=20, label='Attack')
            plt.title('Spatial Distribution')
            plt.xlabel('X Position')
            plt.ylabel('Y Position')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'veremi_analysis_overview.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        # 상세 분석 결과 저장
        self.save_analysis_results(output_dir)
    
    def save_analysis_results(self, output_dir: str):
        """분석 결과를 파일로 저장"""
        # 공격자 분석 결과
        attacker_analysis = self.analyze_attackers()
        
        with open(os.path.join(output_dir, 'attacker_analysis.txt'), 'w') as f:
            f.write("VeReMi GroundTruth Analysis Results\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Total messages: {attacker_analysis['total_messages']}\n")
            f.write(f"Attack messages: {attacker_analysis['attack_messages']}\n")
            f.write(f"Normal messages: {attacker_analysis['normal_messages']}\n")
            f.write(f"Attack ratio: {attacker_analysis['attack_messages'] / attacker_analysis['total_messages'] * 100:.2f}%\n\n")
            
            f.write("Attacker types distribution:\n")
            for attacker_type, count in attacker_analysis['attacker_types'].most_common():
                f.write(f"  Type {attacker_type}: {count} messages ({count/attacker_analysis['attack_messages']*100:.1f}%)\n")
            
            f.write(f"\nUnique attackers: {len(attacker_analysis['attacker_senders'])}\n")
            f.write("Top attackers by message count:\n")
            for sender, count in attacker_analysis['attacker_senders'].most_common(20):
                f.write(f"  Sender {sender}: {count} attack messages\n")
        
        # 공격자 매핑 저장
        attacker_mapping = self.create_attacker_mapping()
        with open(os.path.join(output_dir, 'attacker_mapping.json'), 'w') as f:
            json.dump(attacker_mapping, f, indent=2)
        
        # 시간적 패턴 데이터 저장
        df_temporal = self.analyze_temporal_patterns()
        df_temporal.to_csv(os.path.join(output_dir, 'temporal_patterns.csv'), index=False)
        
        # 공간적 패턴 데이터 저장
        df_spatial = self.analyze_spatial_patterns()
        if not df_spatial.empty:
            df_spatial.to_csv(os.path.join(output_dir, 'spatial_patterns.csv'), index=False)
        
        print(f"Analysis results saved to: {output_dir}")

def main():
    """메인 실행 함수"""
    print("VeReMi GroundTruth Analyzer")
    print("=" * 50)
    
    # GroundTruth 파일 경로
    ground_truth_path = "VeReMi_Data/results/GroundTruthJSONlog.json"
    
    # 분석기 초기화
    analyzer = VeReMiGroundTruthAnalyzer(ground_truth_path)
    
    # 기본 분석 수행
    attacker_analysis = analyzer.analyze_attackers()
    
    # 공격자 매핑 생성
    attacker_mapping = analyzer.create_attacker_mapping()
    
    # 시간적 패턴 분석
    df_temporal = analyzer.analyze_temporal_patterns()
    
    # 공간적 패턴 분석
    df_spatial = analyzer.analyze_spatial_patterns()
    
    # 결과 시각화 및 저장
    analyzer.visualize_analysis()
    
    print("\nAnalysis completed!")
    print("Check the 'veremi_analysis' directory for detailed results.")

if __name__ == "__main__":
    main()
