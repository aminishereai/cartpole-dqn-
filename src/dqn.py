import torch
import torch.nn as nn

# defining the DQN

HIDDEN_DIM = 100
class DQN(nn.Module):
  def __init__(self , state_dim = 4 , action_dim = 2):
    super().__init__()
    self.ffn = nn.Sequential(

        nn.Linear(state_dim , HIDDEN_DIM),
        nn.ReLU(),
        nn.Linear(HIDDEN_DIM , HIDDEN_DIM),
        nn.ReLU(),
        nn.Linear(HIDDEN_DIM , action_dim)

    )

  def forward(self , x):
    return self.ffn(x)

