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


cartpole-dqn/
│
├── README.md                # Project overview, setup, usage instructions
├── requirements.txt         # Python dependencies (torch, gymnasium, matplotlib, etc.)
├── .gitignore               # Ignore checkpoints, videos, etc.
│
├── src/                     # All source code
│   ├── __init__.py
│   ├── dqn.py               # DQN network definition
│   ├── replay_buffer.py     # Replay buffer implementation
│   ├── train.py             # Training loop
│   ├── test.py              # Evaluation / rendering loop
│   └── utils.py             # Helper functions (plotting, epsilon schedule, etc.)
│
├── notebooks/               # Jupyter/Colab notebooks
│   └── DQN_CartPole.ipynb   # Your exploratory notebook
│
├── results/                 # Logs, plots, saved models
│   ├── models/              # Saved checkpoints (.pth files)
│   ├── videos/              # Recorded agent gameplay
│   └── plots/               # Reward curves, training metrics
│
└── LICENSE                  # License file (MIT, Apache 2.0, etc.)

---

## ⚙️ Installation
Clone the repo and install dependencies:

```bash
git clone https://github.com/aminishereai/cartpole-dqn-.git
cd cartpole-dqn-
pip install -r requirements.txt

