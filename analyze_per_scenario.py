# analyze_per_scenario.py

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load saved arrays
errors = np.load("z_data/recon_errors.npy")
labels = np.load("z_data/true_labels.npy")
scenarios = np.load("z_data/scenario_ids.npy")

df = pd.DataFrame({
    "scenario_id": scenarios,
    "recon_error": errors,
    "is_attacker": labels
})

# Plot per scenario
unique_scenarios = sorted(df["scenario_id"].unique())
for scenario in unique_scenarios:
    subset = df[df["scenario_id"] == scenario]
    plt.figure(figsize=(10, 5))
    sns.histplot(data=subset, x="recon_error", hue="is_attacker", bins=80, palette={0: "blue", 1: "red"}, kde=True, stat="density")
    plt.axvline(np.percentile(errors, 85), linestyle="--", color="black", label="Threshold (85%)")
    plt.title(f"Reconstruction Error - Scenario: {scenario}")
    plt.xlabel("MSE")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"z_results/recon_error_{scenario}.png")
    plt.close()
