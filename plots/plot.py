import os
import pandas as pd
import matplotlib.pyplot as plt

# Define absolute paths to the CSV files
rl_only_path = "/Users/admin/Desktop/Sem4/DeepRL/tan_assign_3/policy/exp_local/2025.03.28_rl/174522/eval.csv"
sac_rl_path = "/Users/admin/Desktop/Sem4/DeepRL/tan_assign_3/policy/exp_local/2025.03.28_bcrl/170832/eval.csv"

# Load data into DataFrames
df_rl = pd.read_csv(rl_only_path)
df_sac = pd.read_csv(sac_rl_path)

# Optionally, print columns to check data structure
print("RL Only columns:", df_rl.columns.tolist())
print("SAC RL columns:", df_sac.columns.tolist())

# Create a plot comparing episode rewards over frames for both approaches.
plt.figure(figsize=(10, 6))

# Plot RL-only curve: light green with circle markers, solid line
plt.plot(df_rl["frame"], df_rl["episode_reward"],
         linestyle="-", marker="o", markersize=6, color="lightgreen", label="RL")

# Plot SAC RL curve: orange with triangle markers, dashed line
plt.plot(df_sac["frame"], df_sac["episode_reward"],
         linestyle="--", marker="^", markersize=6, color="orange", label="BCRL")



plt.xlabel("Frame Count")
plt.ylabel("Episode Reward")
plt.title("Q2. Evaluation curves: RL vs BCRL")
plt.legend(loc="best")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
