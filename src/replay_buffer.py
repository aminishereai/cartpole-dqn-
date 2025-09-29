import torch


class ReplayBuffer:
  def __init__(self , max_size = 1000):
    super().__init__()
    self. capacity = max_size
    self.storage = []
    self.position = 0

  def add(self , state , action , reward , next_state , done):
    transition = (state , action , reward , next_state , done)

    if len(self.storage) < self.capacity :
      self.storage.append(transition)
    else :
      self.storage[self.position]  = transition

    self.position = (self.position + 1) % self.capacity

  def sample(self, batch_size):
      # 1. Pick random indices
      ix = torch.randint(0, len(self.storage), (batch_size,))

      # 2. Gather transitions
      mini_batch = [self.storage[i] for i in ix]

      # 3. Unzip into components
      states, actions, rewards, next_states, dones = zip(*mini_batch)

      # 4. Convert to tensors with correct shapes & dtypes
      states      = torch.stack([torch.tensor(s, dtype=torch.float32) for s in states])
      actions     = torch.tensor(actions, dtype=torch.long)
      rewards     = torch.tensor(rewards, dtype=torch.float32)
      next_states = torch.stack([torch.tensor(s, dtype=torch.float32) for s in next_states])
      dones       = torch.tensor(dones, dtype=torch.float32)

      return states, actions, rewards, next_states, dones
  def __len__(self):
    return len(self.storage)
