import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from typing import List, Literal, Tuple
from Training.config import SCORE_GUIDE

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
        self.hidden = torch.zeros(1, 1, self.hidden_size)  # Initialize hidden state
        self.state = torch.tensor([[1, 1]], dtype=torch.float32)  # Initial state

    def next_move(self, agent_moves: List[int], opponent_moves: List[int]) -> Literal[0, 1]:
        action = self.model.learn_and_predict(agent_moves, opponent_moves, self)
        self.print_model_parameters()
        return action
        
    def print_model_parameters(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print(f"Layer: {name}")
                print(f"Weights/Bias: {param.data}")
                print(f"Shape: {param.shape}")
                print("-" * 30)

class RNNAgent(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNAgent, self).__init__()
        self.hidden_size = hidden_size
        self.rnn1 = nn.RNN(input_size, hidden_size, batch_first=True)
        self.rnn2 = nn.RNN(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.optimizer = optim.Adam(self.parameters(), lr=0.1)
        self.log_probs = []
        self.rewards = []

    def forward(self, x, hidden):
        if x.dim() == 2:
            x = x.unsqueeze(0)

        out, hidden1 = self.rnn1(x, hidden)
        out, hidden2 = self.rnn2(out, hidden1)
        out = self.fc(out[:, -1, :])
        return torch.softmax(out, dim=-1), hidden2

    def learn_and_predict(self, self_moves: List[int], opponent_moves: List[int], bot: Bot):
        # For round 1
        if len(self_moves) == 0:
            bot.hidden = torch.zeros(1, 1, self.hidden_size)  # Reset hidden state
            bot.state = torch.tensor([[1, 1]], dtype=torch.float32)  # Initial state (Round 0)
            
            # Reset logs for new episode
            self.log_probs = []
            self.rewards = []

            # Determine action without creating gradients
            with torch.no_grad():
                action_probs, bot.hidden = self(bot.state.unsqueeze(0), bot.hidden)
                action_dist = torch.distributions.Categorical(action_probs)
                action = action_dist.sample()
                
            return action.item()
        
        # For all next rounds
        bot.state = torch.tensor([[self_moves[-1], opponent_moves[-1]]], dtype=torch.float32)
        
        # Learning phase - completely separated from the prediction phase
        if len(self_moves) > 1:  # Only learn after we have some history
            # Create fresh tensors with no gradient history
            states = []
            actions = []
            rewards = []
            
            # Collect recent history (last few moves only)
            # Make sure to not go beyond the beginning of the list
            history_length = min(5, len(self_moves) - 1)  # -1 because we need one previous state too
            
            for i in range(1, history_length + 1):
                idx = -i
                if abs(idx) <= len(self_moves) - 1:  # Check if we can access previous state
                    prev_state = torch.tensor([[self_moves[idx-1], opponent_moves[idx-1]]], dtype=torch.float32)
                    states.append(prev_state)
                    actions.append(self_moves[idx])
                    rewards.append(SCORE_GUIDE[(self_moves[idx], opponent_moves[idx])][0])
            
            # Only continue if we have collected any history
            if states:
                # Reverse to get chronological order
                states.reverse()
                actions.reverse()
                rewards.reverse()
                
                # Calculate discounted returns
                discounted_returns = []
                G = 0
                gamma = 0.9
                for r in reversed(rewards):
                    G = r + gamma * G
                    discounted_returns.insert(0, G)
                
                # Learn from this batch
                self.optimizer.zero_grad()
                batch_loss = 0
                
                # Create a separate hidden state for training
                hidden = torch.zeros(1, 1, self.hidden_size)
                
                for i, (state, action, return_val) in enumerate(zip(states, actions, discounted_returns)):
                    # Forward pass with fresh computation graph
                    action_probs, hidden = self(state.unsqueeze(0), hidden)
                    action_dist = torch.distributions.Categorical(action_probs)
                    log_prob = action_dist.log_prob(torch.tensor(action))
                    batch_loss = batch_loss - log_prob * return_val
                
                # Backprop only after full batch is processed
                batch_loss.backward()
                self.optimizer.step()
        
        # Prediction phase - completely separated from the learning phase
        with torch.no_grad():
            action_probs, bot.hidden = self(bot.state.unsqueeze(0), bot.hidden)
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample()
        
        return action.item()