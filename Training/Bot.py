import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from typing import List, Literal, Tuple
from Training.config import SCORE_GUIDE
import random

class Bot():
    def __init__(self, name="RNN Bot"):
        self.name = name
        self.input_size = 2  # [self_action, opponent_moves]
        self.hidden_size = 16
        self.output_size = 2  # [Cooperate, Defect]
        self.learning_rate = 0.01
        self.num_episodes = 200  # Number of rounds
        self.model = RNNAgent(self.input_size, self.hidden_size, self.output_size)
        self.hidden = torch.zeros(1, 1, self.hidden_size)  # Initialize hidden state
        self.state = torch.tensor([[1, 1]], dtype=torch.float32)  # Initial state
        self.memory = []  # For storing experiences
        self.online_learning = True  # Flag to enable/disable learning
    
    def next_move(self, agent_moves: List[int], opponent_moves: List[int]) -> Literal[0, 1]:
        # Store experience for learning
        if len(agent_moves) > 0 and len(opponent_moves) > 0:
            # Previous state, action and reward
            prev_state = [agent_moves[-2], opponent_moves[-2]] if len(agent_moves) > 1 else [1, 1]
            action = agent_moves[-1]
            
            # Calculate reward based on the prisoner's dilemma payoff matrix
            reward = SCORE_GUIDE[(action, opponent_moves[-1])][0]
                
            # Store the experience
            self.memory.append((prev_state, action, reward))
            
            # Learn from experience if online learning is enabled
            if self.online_learning and len(self.memory) >= 5:
                self.learn_from_memory()
                
        # self.print_model_parameters()
        # Get next action (using the predict method which doesn't create computational graphs)
        return self.model.predict(agent_moves, opponent_moves, self)
    
    def learn_from_memory(self):
        # Only keep the most recent experiences with emphasis on newer ones
        if len(self.memory) > 100:
            self.memory = self.memory[-50:]  # Keep fewer, more recent experiences
            
        # Sample a batch from memory with higher probability for recent experiences
        batch_size = min(len(self.memory), 16)  # Smaller batch for quicker adaptation
        recent_weight = 0.7  # Weight towards recent experiences
        
        # Weighted sampling (recent experiences have higher probability)
        if len(self.memory) > batch_size:
            recent_idx = max(int(batch_size * recent_weight), 1)
            recent_batch = self.memory[-recent_idx:]
            old_batch = random.sample(self.memory[:-recent_idx], batch_size - recent_idx)
            batch = recent_batch + old_batch
        else:
            batch = self.memory
        
        # Prepare batch data
        states = []
        actions = []
        rewards = []
        
        for state, action, reward in batch:
            states.append(torch.tensor([state], dtype=torch.float32))
            actions.append(action)
            rewards.append(reward)
        
        # Calculate discounted returns
        discounted_returns = []
        G = 0
        gamma = 0.9
        for r in reversed(rewards):
            G = r + gamma * G
            discounted_returns.insert(0, G)
            
        # Learn from this batch
        self.model.optimizer.zero_grad()
        batch_loss = 0
        hidden = torch.zeros(1, 1, self.hidden_size)
        
        for i, (state, action, return_val) in enumerate(zip(states, actions, discounted_returns)):
            # Forward pass with fresh computation graph
            action_probs, hidden = self.model(state.unsqueeze(0), hidden)
            action_dist = torch.distributions.Categorical(action_probs)
            log_prob = action_dist.log_prob(torch.tensor(action))
            batch_loss = batch_loss - log_prob * return_val
        
        # Backprop
        batch_loss.backward()
        self.model.optimizer.step()
    
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
        self.optimizer = optim.Adam(self.parameters(), lr=0.2)

    def forward(self, x, hidden):
        if x.dim() == 2:
            x = x.unsqueeze(0)

        out, hidden1 = self.rnn1(x, hidden)
        out, hidden2 = self.rnn2(out, hidden1)
        out = self.fc(out[:, -1, :])
        return torch.softmax(out, dim=-1), hidden2

    def predict(self, self_moves: List[int], opponent_moves: List[int], bot: Bot):
        # For round 1
        if len(self_moves) == 0:
            bot.hidden = torch.zeros(1, 1, self.hidden_size)  # Reset hidden state
            bot.state = torch.tensor([[1, 1]], dtype=torch.float32)  # Initial state
            
        else:
            # For subsequent rounds
            bot.state = torch.tensor([[self_moves[-1], opponent_moves[-1]]], dtype=torch.float32)
        
        # Prediction only - no learning during prediction
        with torch.no_grad():
            action_probs, bot.hidden = self(bot.state.unsqueeze(0), bot.hidden)
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample()
        
        return action.item()