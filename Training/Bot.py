import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class Bot():
    def __init__(self, name):
        # Hyperparameters
        self.name = name
        self.input_size = 2  # [self_action, opponent_action]
        self.hidden_size = 16
        self.output_size = 2  # [Cooperate, Defect]
        self.learning_rate = 0.01
        self.num_episodes = 1000  # Number of rounds
        self.model = RNNAgent(self.input_size, self.hidden_size, self.output_size)

    def predict(self, state):
        state = state.unsqueeze(0)  # Shape: (1, 1, 2) # what
        
        # Get action probabilities and update hidden state
        action_probs, self.hidden = self.model(state, self.hidden)
        
        # Sample an action from the probability distribution
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample().item()

        return action
        
        


# Define the RNN-based agent
class RNNAgent(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNAgent, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        out, hidden = self.rnn(x, hidden)
        out = self.fc(out)
        return torch.softmax(out, dim=-1), hidden
    
    def predict(self, x, hidden):
        # Not sure what parameters to pass through
        return self.forward(x, hidden)
