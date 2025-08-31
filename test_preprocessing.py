import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 파일 로드
train_X = np.load("z_data/train_X.npy")
train_y = np.load("z_data/train_y.npy")
test_X = np.load("z_data/test_X.npy")
test_y = np.load("z_data/test_y.npy")

print("✅ Loaded Shapes")
print("train_X:", train_X.shape)
print("train_y:", train_y.shape)
print("test_X :", test_X.shape)
print("test_y :", test_y.shape)

# y 값 확인
print("\n✅ Label Distribution")
print("train_y unique:", np.unique(train_y))
print("test_y unique :", np.unique(test_y, return_counts=True))

# 시퀀스 하나 확인
sample = test_X[0]
print("\n✅ Sample sequence shape:", sample.shape)

# 시각화 (특성별 시간 흐름)
feature_names = ["pos_x", "pos_y", "pos_z", "speed", "heading", "curvature"]

plt.figure(figsize=(12, 6))
for i in range(train_X.shape[2]):
    plt.plot(sample[:, i], label=feature_names[i])
plt.title("📊 Sample Test Sequence Feature Over Time")
plt.xlabel("Time Step")
plt.ylabel("Value (Standardized)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
