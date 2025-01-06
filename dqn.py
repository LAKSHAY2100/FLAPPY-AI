import torch
from torch import nn
import torch.nn.functional as F

class DQN(nn.Module):
    # input_dim == states
    # output_dim == actions
    def __init__(self, state_dim, action_dim,hidden_dim=256):
        super(DQN, self).__init__()

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
if __name__ == "__main__":
    state_dim = 12  # Example state dimension
    action_dim = 2  # Example action dimension
    model = DQN(state_dim, action_dim)
    # Example input tensor
    state = torch.rand(10, state_dim)
    # Forward pass through the model

    output = model(state)
    # The output is a tensor of shape (10, action_dim) containing the predicted
    # action values for each of the 10 states in the input tensor.
    print("Model output:", output)

