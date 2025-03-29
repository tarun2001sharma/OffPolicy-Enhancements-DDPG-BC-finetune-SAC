import pandas as pd
import matplotlib.pyplot as plt

def plot_train_reward(csv_file="/Users/admin/Desktop/Sem4/DeepRL/assignment_3/assignment 3/policy/exp_local/2025.03.28_rl/031117/train.csv"):
    df = pd.read_csv(csv_file)
    
    plt.figure(figsize=(10, 6))
    plt.plot(df["frame"], df["episode_reward"], marker='o', label="Training Episode Reward")
    plt.xlabel("Frame")
    plt.ylabel("Episode Reward")
    plt.title("Training Episode Reward vs. Frame")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_train_reward("/Users/admin/Desktop/Sem4/DeepRL/assignment_3/assignment 3/policy/exp_local/2025.03.28_rl/031117/train.csv")
