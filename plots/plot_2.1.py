import os
import pandas as pd
import matplotlib.pyplot as plt




# Set the file paths (adjust if necessary)
train_csv_path = os.path.join(os.getcwd(), "/Users/admin/Desktop/Sem4/DeepRL/assignment_3/assignment 3/policy/exp_local/2025.03.28_rl/031117/train.csv")
trainbcrl_csv_path = os.path.join(os.getcwd(), "/Users/admin/Desktop/Sem4/DeepRL/assignment_3/assignment 3/policy/exp_local/2025.03.28_rl/185439/train.csv")

# Load the CSV files into pandas DataFrames
train_df = pd.read_csv(train_csv_path)
trainbcrl_df = pd.read_csv(trainbcrl_csv_path)

# Print out the column names to verify the structure (optional)
print("Train RL only CSV columns:", train_df.columns.tolist())
print("Train SAC RL CSV columns:", trainbcrl_df.columns.tolist())

plt.figure(figsize=(12, 6))

# Use 'frame' as the x-axis instead of 'global_frame'
plt.plot(train_df["frame"], train_df["episode_reward"], 
         marker="o", label="Training RL Only Episode Reward")

plt.plot(trainbcrl_df["frame"], trainbcrl_df["episode_reward"], 
         marker="x", linestyle="--", label="Training SAC RL Episode Reward")

plt.xlabel("Frame")
plt.ylabel("Episode Reward")
plt.title("Training Curves - RL only vs SAC RL")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
