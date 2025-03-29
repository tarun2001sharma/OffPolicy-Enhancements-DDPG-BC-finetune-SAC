import os
import pandas as pd
import matplotlib.pyplot as plt




# Set the file paths (adjust if necessary)
eval_csv_path = os.path.join(os.getcwd(), "/Users/admin/Desktop/Sem4/DeepRL/assignment_3/assignment 3/policy/exp_local/2025.03.28_rl/031117/eval.csv")
eval2_csv_path = os.path.join(os.getcwd(), "/Users/admin/Desktop/Sem4/DeepRL/assignment_3/assignment 3/policy/exp_local/2025.03.28_rl/185439/eval.csv")

# Load the CSV files into pandas DataFrames
eval_df = pd.read_csv(eval_csv_path)
eval2_df = pd.read_csv(eval2_csv_path)

# Print out the column names to verify the structure (optional)
print("eval RL only CSV columns:", eval_df.columns.tolist())
print("eval BSAC RL CSV columns:", eval2_df.columns.tolist())

plt.figure(figsize=(12, 6))

# Use 'frame' as the x-axis instead of 'global_frame'
plt.plot(eval_df["frame"], eval_df["episode_reward"], 
         marker="o", label="Training RL Only Episode Reward")

plt.plot(eval2_df["frame"], eval2_df["episode_reward"], 
         marker="x", linestyle="--", label="Training SAC RL Episode Reward")

plt.xlabel("Frame")
plt.ylabel("Episode Reward")
plt.title("Training Curves - RL only vs SAC RL")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
