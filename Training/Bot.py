import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from typing import List, Literal, Tuple
from Training.config import SCORE_GUIDE
import random

class Bot():
    def __init__(self, name="RNN Bot", output: bool = True, verbose: bool = False):
        # Constants
        self.name = name
        self.input_size = 2  # [self_action, opponent_moves]
        self.hidden_size = 16
        self.output_size = 2  # [Cooperate, Defect]

        # Parameters
        self.learning_rate = 0.01  # Learning rate during gym training
        self.num_episodes = 200  # Number of rounds

        # Data structures
        self.model = RNNAgent(self.input_size, self.hidden_size, self.output_size)
        self.hidden = torch.zeros(1, 1, self.hidden_size)  # Initialize hidden state
        self.state = torch.tensor([[1, 1]], dtype=torch.float32)  # Initial state
        self.trained_state = None  # For storing the post-training weights
        self.memory = []  # For storing experiences

        # Flags
        self.online_learning = True  # Flag to enable/disable learning
        self.output = output  # Flag for debug output
        self.verbose = verbose  # Flag for verbose output
    
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
        # Calculate action probabilities based on the current state
        if self.output:
            with torch.no_grad():
                action_probs, _ = self.model(self.state.unsqueeze(0), self.hidden)
                print(f"Action probabilities: {action_probs.squeeze().tolist()}")
        # Get next action (using the predict method which doesn't create computational graphs)
        return self.model.predict(agent_moves, opponent_moves, self)
    
    def learn_from_memory(self):
        """Learn from recent experiences without explicit heuristics"""        
        # Use recent experiences to adapt to current opponent
        experiences = self.memory[-15:]
        
        # Unpack experiences
        prev_states = [exp[0] for exp in experiences]
        actions = [exp[1] for exp in experiences]
        rewards = [exp[2] for exp in experiences]
        
        # Create optimizer with consistent learning rate
        optimizer = optim.Adam(self.model.parameters(), lr=0.1)
        
        # Calculate returns with reasonable discount factor
        gamma = 0.9
        returns = []
        R = 0
        for reward in reversed(rewards):
            R = reward + gamma * R
            returns.insert(0, R)
        
        # Convert to tensors
        returns_tensor = torch.tensor(returns, dtype=torch.float32)
        
        # Normalize returns for stable learning
        if len(returns) > 1 and returns_tensor.std() > 0:
            returns_tensor = (returns_tensor - returns_tensor.mean()) / (returns_tensor.std() + 1e-5)
        
        # Reset hidden state
        hidden = torch.zeros(1, 1, self.hidden_size)
        
        # Compute log probs
        log_probs = []
        entropy = 0
        
        for i in range(len(prev_states)):
            state = torch.tensor([prev_states[i]], dtype=torch.float32).unsqueeze(0)
            action_probs, hidden = self.model(state, hidden)
            dist = torch.distributions.Categorical(action_probs.squeeze())
            
            # Convert action to tensor if needed
            action_tensor = actions[i] if isinstance(actions[i], torch.Tensor) else torch.tensor(actions[i], dtype=torch.long)
            log_prob = dist.log_prob(action_tensor)
            log_probs.append(log_prob)
            
            # Add small entropy bonus to encourage exploration
            entropy += dist.entropy()
        
        # Stack log probs
        log_probs = torch.stack(log_probs)
        
        # Policy gradient loss with small entropy regularization
        policy_loss = -torch.sum(log_probs * returns_tensor) - 0.01 * entropy
        
        # Perform optimization
        optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        optimizer.step()
        
        # Output debug information
        if self.verbose:
            print(f"Online learning from {len(experiences)} experiences")
            print(f"Average reward: {sum(rewards) / len(rewards):.2f}")
            print(f"Policy loss: {policy_loss.item():.4f}")
        elif self.output:
            print(f"Learning from {len(experiences)} recent moves")
        
        # Keep only recent memory to focus on current opponent
        self.memory = self.memory[-20:]
        
    
    def save_trained_state(self):
        """Save the current model weights as the reference post-training state"""
        self.trained_state = {key: val.clone() for key, val in self.model.state_dict().items()}

    def reset_to_trained_state(self):
        """Reset model weights to the saved post-training state"""
        if self.trained_state is not None:
            self.model.load_state_dict(self.trained_state)
            # Also reset the hidden state
            self.hidden = torch.zeros(1, 1, self.hidden_size)
        else:
            raise ValueError("No trained state found. Please train the model first.")

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