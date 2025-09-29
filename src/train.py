import torch
import torch.nn.functional as F
import gymnasium as gym
from dqn import DQN
from replay_buffer import ReplayBuffer
from utils import plot_rewards

# Hyperparameters
capacity = 10000
lr = 1e-3
episodes = 500
max_episode_len = 200
min_threshold = 1000
gamma = 0.99
batch_size = 64
epsilon_decay = 0.995
epsilon_min = 0.05
tau = 0.005

def train():
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    q_net = DQN(state_dim, action_dim)
    target_net = DQN(state_dim, action_dim)
    target_net.load_state_dict(q_net.state_dict())

    buffer = ReplayBuffer(max_size=capacity)
    optimizer = torch.optim.Adam(q_net.parameters(), lr=lr)

    epsilon = 1.0
    rewards_per_episode = []

    for episode in range(episodes):
        state, _ = env.reset()
        state = torch.as_tensor(state, dtype=torch.float32)
        total_reward = 0

        for step in range(max_episode_len):
            # Epsilon-greedy
            if torch.rand(1).item() < epsilon:
                action = env.action_space.sample()
            else:
                q_values = q_net(state.unsqueeze(0))
                action = q_values.argmax(-1).item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = torch.as_tensor(next_state, dtype=torch.float32)

            buffer.add(state, action, reward, next_state, done)
            total_reward += reward

            if len(buffer) >= min_threshold:
                s_b, a_b, r_b, n_s_b, d_b = buffer.sample(batch_size)
                preds = q_net(s_b)[torch.arange(0, s_b.shape[0]), a_b]
                with torch.no_grad():
                    y = r_b + gamma * target_net(n_s_b).max(-1)[0] * (1 - d_b)
                loss = F.mse_loss(preds, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            state = next_state
            if done:
                break

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        rewards_per_episode.append(total_reward)
        print(f"Episode {episode+1} | Epsilon {epsilon:.3f} | Reward {total_reward}")

        # Soft update
        with torch.no_grad():
            for p1, p2 in zip(q_net.parameters(), target_net.parameters()):
                p2.data.copy_(tau * p2.data + (1 - tau) * p1.data)

    plot_rewards(rewards_per_episode, window=20, save_path="results/plots/reward_curve.png")
    torch.save(q_net.state_dict(), "results/models/dqn_cartpole.pth")

if __name__ == "__main__":
    train()
