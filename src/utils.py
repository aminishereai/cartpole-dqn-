import matplotlib.pyplot as plt
import pandas as pd

def plot_rewards(rewards, window=20, save_path=None):
    plt.figure(figsize=(10,6))
    plt.plot(rewards, label="Reward per Episode", alpha=0.6)
    plt.plot(pd.Series(rewards).rolling(window).mean(),
             label=f"Moving Average ({window})", color="red")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("CartPole DQN Training Progress")
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    plt.show()
