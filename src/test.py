import torch
import gymnasium as gym
import imageio
from dqn import DQN

def test(model_path="results/models/dqn_cartpole.pth", episodes=3, max_steps=500, save_video=True):
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    q_net = DQN(state_dim, action_dim)
    q_net.load_state_dict(torch.load(model_path))
    q_net.eval()

    for ep in range(episodes):
        state, _ = env.reset()
        state = torch.as_tensor(state, dtype=torch.float32)
        total_reward = 0
        frames = []

        for step in range(max_steps):
            q_values = q_net(state.unsqueeze(0))
            action = q_values.argmax(-1).item()
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

            frames.append(env.render())
            state = torch.as_tensor(next_state, dtype=torch.float32)
            if done:
                break

        print(f"Test Episode {ep+1}: Reward = {total_reward}")

        if save_video:
            filename = f"results/videos/cartpole_test_ep{ep+1}.mp4"
            imageio.mimsave(filename, frames, fps=30)
            print(f"Saved video: {filename}")

if __name__ == "__main__":
    test()
