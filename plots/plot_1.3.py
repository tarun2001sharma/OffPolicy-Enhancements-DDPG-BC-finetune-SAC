import pandas as pd
import matplotlib.pyplot as plt

def plot_actor_loss(csv_file="train.csv"):
    df = pd.read_csv(csv_file)

    plt.figure(figsize=(10, 6))
    plt.plot(df["frame"], df["actor_loss"], label="Actor Loss")
    plt.xlabel("Frame")
    plt.ylabel("Loss")
    plt.title("Actor Loss vs. Frame")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_actor_loss("/Users/admin/Desktop/Sem4/DeepRL/assignment_3/assignment 3/policy/exp_local/2025.03.28_rl/031117/train.csv")
