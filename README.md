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
cartpole-dqn/ │ ├── src/ │ ├── dqn.py # DQN network definition │ ├── replay_buffer.py # Replay buffer implementation │ ├── train.py # Training loop │ ├── test.py # Evaluation / rendering │ └── utils.py # Helper functions (plotting, epsilon schedule) │ ├── notebooks/ │ └── DQN_CartPole.ipynb # Exploratory notebook │ ├── results/ │ ├── models/ # Saved checkpoints │ ├── videos/ # Recorded gameplay │ └── plots/ # Reward curves │ ├── requirements.txt # Dependencies ├── LICENSE # MIT License └── README.md # This file


---

## ⚙️ Installation
Clone the repo and install dependencies:

```bash
git clone https://github.com/aminishereai/cartpole-dqn-.git
cd cartpole-dqn-
pip install -r requirements.txt

