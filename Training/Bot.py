import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Literal

class Bot():
    def __init__(self, name):
        # Hyperparameters
        self.name = name
        self.input_size = 2  # [self_action, opponent_action]
        self.hidden_size = 16
        self.output_size = 2  # [Cooperate, Defect]
        self.learning_rate = 0.01
        self.num_episodes = 200  # Number of rounds
        self.model = RNNAgent(self.input_size, self.hidden_size, self.output_size)
        self.state = torch.tensor([[0, 0]], dtype=torch.float32)  # Initial state
        self.hidden = torch.zeros(1, self.hidden_size)  # Initialize hidden state

    def next_move(self, agent_moves: List[int], opponent_moves: List[int]) -> Literal[0, 1]:
        action = self.model.predict(agent_moves, opponent_moves)

        if len(agent_moves) > 0:
            self.model.learn(agent_moves, opponent_moves, self.hidden, self.state)
        return action

# Define the RNN-based agent
class RNNAgent(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNAgent, self).__init__()
        self.hidden_size = hidden_size
        self.rnn1 = nn.RNN(input_size, hidden_size, batch_first=True)
        self.rnn2 = nn.RNN(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.log_probs = []

    def forward(self, x, hidden):
        # Ensure hidden state is 3-dimensional
        if hidden.dim() == 2:
            hidden = hidden.unsqueeze(0)
        
        # Pass input through the first RNN layer
        out, hidden1 = self.rnn1(x, hidden)
        
        # Pass the output of the first RNN layer into the second RNN layer
        out, hidden2 = self.rnn2(out, hidden1)
        
        # Take the output of the last time step and pass it through the fully connected layer
        out = self.fc(out[:, -1, :])
        
        # Apply softmax to get action probabilities
        return torch.softmax(out, dim=-1), hidden2

    def predict(self, agent_moves: List[int], opponent_moves: List[int]) -> Literal[0, 1]:
        # Ensure the input lists are of the same length
        assert len(agent_moves) == len(opponent_moves), "Agent and opponent moves must be of the same length."

        # Convert the moves into a tensor
        if len(agent_moves) == 0:
            moves = torch.tensor([[1, 1]], dtype=torch.float32).unsqueeze(0)
        else:
            moves = torch.tensor([[agent_moves[-1], opponent_moves[-1]]], dtype=torch.float32).unsqueeze(0)  # Shape: (1, 1, 2)

        # Initialize hidden state
        hidden = torch.zeros(1, self.hidden_size)

        # Pass the input through the model
        action_probs, _ = self.forward(moves, hidden.unsqueeze(0))

        # Choose the action with the highest probability
        predicted_action = torch.argmax(action_probs, dim=-1).item()

        return predicted_action
    
    def learn(self, agent_moves: List[int], opponent_moves: List[int], state, hidden):
        # Ensure hidden state is 3-dimensional
        if hidden.dim() == 2:
            hidden = hidden.unsqueeze(0)
        
        # Convert the state to the correct input size
        state = state.unsqueeze(0)  # Shape: (1, 1, 2)

        # Determine reward
        if agent_moves[-1] == 1 and opponent_moves[-1] == 1:
            reward = torch.tensor([3])  # Mutual cooperation
        elif agent_moves[-1] == 1 and opponent_moves[-1] == 0:
            reward = torch.tensor([0])  # Sucker's payoff
        elif agent_moves[-1] == 0 and opponent_moves[-1] == 1:
            reward = torch.tensor([5])  # Temptation to defect
        else:
            reward = torch.tensor([1])  # Mutual defection

        # Ensure state is 3-dimensional
        state = state.unsqueeze(0)  # Shape: (1, 1, 2)

        action_probs, hidden = self(state, hidden)
        action_dist = torch.distributions.Categorical(action_probs)

        log_prob = action_dist.log_prob(torch.tensor(agent_moves[-1]))
        self.log_probs.append(log_prob)

        gamma = 0.9  # Discount factor

        discounted_returns = gamma * reward

        discounted_returns = torch.tensor(discounted_returns, dtype=torch.float32)
        baseline = discounted_returns.mean()  # Baseline is the mean of returns

        optimizer = optim.Adam(self.parameters(), lr=0.01)
        optimizer.zero_grad()
        # Calculate policy loss using discounted returns minus baseline
        policy_loss = -sum([log_prob * (return_t - baseline) for log_prob, return_t in zip(self.log_probs, discounted_returns)])
        policy_loss.backward()
        optimizer.step()