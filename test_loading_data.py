import base64
import io
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

import os
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from pathlib import Path

def check_feature_distributions(df, dataset_name=""):
    print(f"\n📊 {dataset_name} Feature Distribution & Outlier Check")

    results = {}

    # 1. Speed 분포 확인
    if "speed" in df.columns:
        speed_stats = df["speed"].describe()
        print("Speed 통계:\n", speed_stats)
        # 비정상 속도 탐지 (예: 80 m/s ≈ 288 km/h 이상은 아웃라이어로 간주)
        high_speed = df[df["speed"] > 80]
        results["high_speed_count"] = len(high_speed)
        if len(high_speed) > 0:
            print(f"⚠️ 비정상 속도 샘플 발견: {len(high_speed)}개 (80 m/s 이상)")
        sns.histplot(df["speed"], bins=50, kde=True)
        plt.title(f"{dataset_name} - Speed Distribution")
        plt.xlabel("Speed (m/s)")
        plt.show()

    # 2. 좌표 범위 확인
    if "pos_x" in df.columns and "pos_y" in df.columns:
        print(f"좌표 범위: X({df['pos_x'].min():.1f} ~ {df['pos_x'].max():.1f}), "
              f"Y({df['pos_y'].min():.1f} ~ {df['pos_y'].max():.1f})")
        sns.scatterplot(x="pos_x", y="pos_y", data=df.sample(min(5000, len(df))))
        plt.title(f"{dataset_name} - Position Scatter (sample)")
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.show()

    # 3. 헤딩(heading) 분포 확인
    if "heading" in df.columns:
        heading_stats = df["heading"].describe()
        print("Heading 통계:\n", heading_stats)
        # 범위 확인 (0~360도 범위 벗어난 값 탐지)
        invalid_heading = df[(df["heading"] < 0) | (df["heading"] > 360)]
        results["invalid_heading_count"] = len(invalid_heading)
        if len(invalid_heading) > 0:
            print(f"⚠️ 잘못된 heading 값 발견: {len(invalid_heading)}개")
        sns.histplot(df["heading"], bins=50, kde=False)
        plt.title(f"{dataset_name} - Heading Distribution")
        plt.xlabel("Heading (degree)")
        plt.show()

    # 🚦 시나리오 및 차량별 데이터 분포 분석
    print("="*100)
    print("시나리오별로 분석")
    print("="*100)

    scenario_station_counts = df.groupby("scenario_id")["station_id"].nunique().sort_values(ascending=False)
    print(scenario_station_counts.head(10))
    scenario_station_counts.plot(kind="bar", figsize=(12,4))

    plt.title("vehicle count by scenario")
    plt.ylabel("vehicle count")
    plt.xlabel("Scenario ID")
    plt.tight_layout()
    plt.show()
    
    station_lengths = df.groupby(["scenario_id", "station_id"]).size().sort_values(ascending=False)
    print(station_lengths.describe())

    plt.figure(figsize=(10, 4))
    sns.histplot(station_lengths, bins=50)
    plt.title("message count by vehicle")
    plt.xlabel("CAM message count")
    plt.show()

    # 데이터가 가장 많은 차량 확인
    longest_seq_id = station_lengths.idxmax()
    scenario, station = longest_seq_id

    # 해당 차량 데이터 시각화
    sub_df = df[(df["scenario_id"] == scenario) & (df["station_id"] == station)].sort_values("timestamp")

    plt.figure(figsize=(10,4))
    plt.plot(sub_df["timestamp"].values, marker='o')
    plt.title(f"⏱ Timestamp flow - scenario={scenario}, station_id={station}")
    plt.xlabel("Index")
    plt.ylabel("Timestamp (s)")
    plt.grid()
    plt.show()

    gap_stats = []

    for (scenario, station), group in df.groupby(["scenario_id", "station_id"]):
        ts = group.sort_values("timestamp")["timestamp"].values
        if len(ts) >= 2:
            gaps = np.diff(ts)
            gap_stats.append({
                "scenario_id": scenario,
                "station_id": station,
                "num_samples": len(ts),
                "mean_gap": np.mean(gaps),
                "std_gap": np.std(gaps),
                "max_gap": np.max(gaps)
            })

    gap_df = pd.DataFrame(gap_stats)
    print(gap_df.describe())

    return results


def compare_attacker_vs_normal_to_html(df, scenario_id, save_dir="reports"):
    os.makedirs(save_dir, exist_ok=True)
    html_path = Path(save_dir) / f"{scenario_id}.html"

    # ---- 📊 기존 시각화 ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Scenario: {scenario_id}", fontsize=16)

    # Speed
    sns.kdeplot(df[df["is_attacker"] == 0]["speed"], label="Normal", fill=True, ax=axes[0, 0])
    sns.kdeplot(df[df["is_attacker"] == 1]["speed"], label="Attacker", fill=True, ax=axes[0, 0], color='orange')
    axes[0, 0].set_title("Speed Distribution")
    axes[0, 0].set_xlabel("Speed (m/s)")
    axes[0, 0].legend()

    # Heading
    sns.histplot(df[df["is_attacker"] == 0]["heading"], bins=50, color="blue", alpha=0.5, label="Normal", ax=axes[0, 1])
    sns.histplot(df[df["is_attacker"] == 1]["heading"], bins=50, color="red", alpha=0.5, label="Attacker", ax=axes[0, 1])
    axes[0, 1].set_title("Heading Distribution")
    axes[0, 1].set_xlabel("Heading (degrees)")
    axes[0, 1].legend()

    # Message count
    lengths = df.groupby(["station_id", "is_attacker"]).size().reset_index(name="message_count")
    sns.boxplot(data=lengths, x="is_attacker", y="message_count", ax=axes[1, 0])
    axes[1, 0].set_xticklabels(["Normal", "Attacker"])
    axes[1, 0].set_title("Message Count per Vehicle")
    axes[1, 0].set_ylabel("Number of CAM Messages")

    # Position
    sample_normal = df[df["is_attacker"] == 0].sample(min(3000, sum(df["is_attacker"] == 0)))
    sample_attacker = df[df["is_attacker"] == 1].sample(min(3000, sum(df["is_attacker"] == 1)))
    axes[1, 1].scatter(sample_normal["pos_x"], sample_normal["pos_y"], label="Normal", alpha=0.5, s=10)
    axes[1, 1].scatter(sample_attacker["pos_x"], sample_attacker["pos_y"], label="Attacker", alpha=0.5, s=10, color='red')
    axes[1, 1].set_title("Position Distribution (Sampled)")
    axes[1, 1].set_xlabel("pos_x")
    axes[1, 1].set_ylabel("pos_y")
    axes[1, 1].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # ---- 이미지 저장 ----
    image_path = Path(save_dir) / f"{scenario_id}.png"
    fig.savefig(image_path)
    plt.close()

    # ---- HTML 생성 ----
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(f"""
        <html>
        <head><title>VeReMi Analysis Report - {scenario_id}</title></head>
        <body>
            <h1>Scenario: {scenario_id}</h1>
            <p><strong>Vehicle Count:</strong> Normal={sum(df['is_attacker']==0)}, Attacker={sum(df['is_attacker']==1)}</p>
            <img src="{scenario_id}.png" width="100%" />
        </body>
        </html>
        """)
    print(f"✅ HTML 리포트 저장됨: {html_path}")


def generate_single_html_report(df_all, output_path="veremi_report.html"):
    html_parts = ["<html><head><title>VeReMi Scenario Analysis</title></head><body>"]
    html_parts.append("<h1>🚗 VeReMi Scenario Analysis Report</h1>")

    for scenario_id in df_all["scenario_id"].unique():
        df = df_all[df_all["scenario_id"] == scenario_id].copy()
        df["heading"] = (df["heading"] + 360) % 360  # normalize

        # Plot
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"Scenario: {scenario_id}", fontsize=16)

        # Speed
        sns.kdeplot(df[df["is_attacker"] == 0]["speed"], label="Normal", fill=True, ax=axes[0, 0])
        sns.kdeplot(df[df["is_attacker"] == 1]["speed"], label="Attacker", fill=True, ax=axes[0, 0], color='orange')
        axes[0, 0].set_title("Speed Distribution")
        axes[0, 0].set_xlabel("Speed (m/s)")
        axes[0, 0].legend()

        # Heading
        sns.histplot(df[df["is_attacker"] == 0]["heading"], bins=50, color="blue", alpha=0.5, label="Normal", ax=axes[0, 1])
        sns.histplot(df[df["is_attacker"] == 1]["heading"], bins=50, color="red", alpha=0.5, label="Attacker", ax=axes[0, 1])
        axes[0, 1].set_title("Heading Distribution")
        axes[0, 1].set_xlabel("Heading (degrees)")
        axes[0, 1].legend()

        # Message count
        lengths = df.groupby(["station_id", "is_attacker"]).size().reset_index(name="message_count")
        sns.boxplot(data=lengths, x="is_attacker", y="message_count", ax=axes[1, 0])
        axes[1, 0].set_xticklabels(["Normal", "Attacker"])
        axes[1, 0].set_title("Message Count per Vehicle")
        axes[1, 0].set_ylabel("Number of CAM Messages")

        # Position
        sample_normal = df[df["is_attacker"] == 0].sample(min(3000, sum(df["is_attacker"] == 0)))
        sample_attacker = df[df["is_attacker"] == 1].sample(min(3000, sum(df["is_attacker"] == 1)))
        axes[1, 1].scatter(sample_normal["pos_x"], sample_normal["pos_y"], label="Normal", alpha=0.5, s=10)
        axes[1, 1].scatter(sample_attacker["pos_x"], sample_attacker["pos_y"], label="Attacker", alpha=0.5, s=10, color='red')
        axes[1, 1].set_title("Position Distribution (Sampled)")
        axes[1, 1].set_xlabel("pos_x")
        axes[1, 1].set_ylabel("pos_y")
        axes[1, 1].legend()

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        # Convert to base64
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode("utf-8")
        plt.close(fig)

        # Add to HTML
        html_parts.append(f"<h2>{scenario_id}</h2>")
        html_parts.append(f'<img src="data:image/png;base64,{img_base64}" width="100%"/>')

    html_parts.append("</body></html>")

    # Save to file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(html_parts))

    print(f"✅ 전체 리포트 저장됨: {output_path}")



if __name__ == "__main__":


    v2aix_df = pd.read_csv("z_test/v2aix_etsi_standardized.csv")
    veremi_df = pd.read_csv("z_test/veremi_etsi_standardized.csv")

    print("V2AIX shape: ", v2aix_df.shape)
    print("Veremi shape: ", veremi_df.shape)

    check_feature_distributions(v2aix_df, "V2AIX")
    check_feature_distributions(veremi_df, "Veremi")

    # html로 공격/정상 비교
    # generate_single_html_report(veremi_df, output_path="veremi_report5.html")
    generate_single_html_report(v2aix_df, output_path="v2aix_report.html")