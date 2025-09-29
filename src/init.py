capacity = 10000
lr =1e-3
episodes = 1000
max_episode_len = 200
min_threshold = 1000
gamma = 0.99
batch_size = 64
epsilon_decay = 0.995
epsilon_min = 0.05
tau = 0.005

# Initializing Environment , Q-net , Target-net , Replay buffer and optimizer
env = gym.make("CartPole-v1", render_mode="rgb_array")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
state = env.reset()


q_net = DQN(state_dim , action_dim)
target_net = DQN(state_dim , action_dim)
target_net.load_state_dict(q_net.state_dict()) # Initalizing networks with exactly same weights



buffer = ReplayBuffer(max_size=capacity)


optimizer = torch.optim.Adam(q_net.parameters() , lr = lr)
