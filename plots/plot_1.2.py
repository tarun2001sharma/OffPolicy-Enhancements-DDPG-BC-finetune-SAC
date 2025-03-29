import pandas as pd
import matplotlib.pyplot as plt

def plot_eval_reward(csv_file="eval.csv"):
    df = pd.read_csv(csv_file)
    
    plt.figure(figsize=(10, 6))
    plt.plot(df["frame"], df["episode_reward"], marker='x', label="Evaluation Episode Reward", linestyle='--')
    plt.xlabel("Frame")
    plt.ylabel("Episode Reward")
    plt.title("Evaluation Episode Reward vs. Frame")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_eval_reward("/Users/admin/Desktop/Sem4/DeepRL/assignment_3/assignment 3/policy/exp_local/2025.03.28_rl/031117/eval.csv")
