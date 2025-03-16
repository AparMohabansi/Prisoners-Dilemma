import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

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

# Hyperparameters
input_size = 2  # [self_action, opponent_action]
hidden_size = 16
output_size = 2  # [Cooperate, Defect]
learning_rate = 0.01
num_episodes = 1000  # Number of rounds

# Initialize agent, optimizer, and loss function
agent = RNNAgent(input_size, hidden_size, output_size)
optimizer = optim.Adam(agent.parameters(), lr=learning_rate)

# Training loop
for episode in range(num_episodes):
    hidden = torch.zeros(1, 1, hidden_size)  # Initialize hidden state
    log_probs = []
    rewards = []

    # Play one episode (multiple rounds)
    for _ in range(10):  # Play 10 rounds per episode
        # Get current state (previous actions)
        if len(log_probs) > 0:
            state = torch.tensor([[self_action, opponent_action]], dtype=torch.float32)
        else:
            state = torch.tensor([[0, 0]], dtype=torch.float32)  # Initial state (cooperate to start)

        # Get action probabilities
        action_probs, hidden = agent(state, hidden)
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        log_probs.append(log_prob)

        # Opponent's action (random for simplicity)
        opponent_action = np.random.randint(0, 2) # can change to user input later

        # Opponent's action (Tit for Tat: copy agent's previous move)
        opponent_action = self_action  # Opponent mirrors agent's previous move

        # Determine reward
        if action.item() == 1 and opponent_action == 1:
            reward = 3  # Mutual cooperation
        elif action.item() == 1 and opponent_action == 0:
            reward = 0  # Sucker's payoff
        elif action.item() == 0 and opponent_action == 1:
            reward = 5  # Temptation to defect
        else:
            reward = 1  # Mutual defection
        rewards.append(reward)

        # Update self_action and opponent_action for next state
        self_action = action.item()

    # Compute cumulative reward
    cumulative_reward = sum(rewards)

    # Update agent
    optimizer.zero_grad()
    policy_loss = -sum([log_prob * reward for log_prob, reward in zip(log_probs, rewards)])
    policy_loss.backward()
    optimizer.step()

    if episode % 100 == 0:
        print(f"Episode {episode}, Cumulative Reward: {cumulative_reward}")