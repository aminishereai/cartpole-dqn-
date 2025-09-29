# CartPole DQN 🎮

An implementation of **Deep Q‑Network (DQN)** to solve the classic **CartPole‑v1** environment from [Gymnasium](https://gymnasium.farama.org/).  
Built with **PyTorch**, this project demonstrates reinforcement learning fundamentals: replay buffer, epsilon‑greedy exploration, target network updates, and training loop.

---

## 🚀 Features
- DQN agent implemented in PyTorch
- Experience Replay Buffer
- Epsilon‑greedy exploration strategy
- Target network with soft/hard updates
- Training loop with reward tracking
- Evaluation script with video recording of the trained agent

---

## 📂 Project Structure

<br>

cartpole-dqn/ │<br>├── README.md # Project overview, setup, usage instructions <br>├── requirements.txt # Python dependencies <br>├── .gitignore # Ignore checkpoints, videos, etc. <br>│ <br>├── src/ # Source code <br>│ <br>├── init.py │ <br>├── dqn.py # DQN network definition <br>│ <br>├── replay_buffer.py # Replay buffer implementation <br>│ <br>├── train.py # Training loop <br>│ <br>├── test.py # Evaluation / rendering loop <br>│ └── utils.py # Helper functions (plotting, epsilon schedule, etc.) <br>│ <br>├── notebooks/ # Jupyter/Colab notebooks <br>│ └── DQN_CartPole.ipynb # Exploratory notebook <br>│ <br>├── results/ # Logs, plots, saved models <br>│ <br>├── models/ # Saved checkpoints (.pth files) <br>│ <br>├── videos/ # Recorded agent gameplay <br>│ └── plots/ # Reward curves, training metrics <br>│ └── LICENSE # License file (MIT, Apache 2.0, etc.)
---

## ⚙️ Installation
Clone the repo and install dependencies:

```bash
git clone https://github.com/aminishereai/cartpole-dqn-.git
cd cartpole-dqn-
pip install -r requirements.txt

