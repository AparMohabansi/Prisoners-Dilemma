from Training.Agent import Agent
from Training.Bot import Bot, RNNAgent
from Training.config import SCORE_GUIDE
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from typing import List

class Gym():
    def __init__(self, agents: List[Agent], bot: Bot):
        self.agents = agents
        self.bot = bot
    
    def RNN_train(self):
        # Initialize agent, optimizer, and loss function
        model = self.bot.model
        optimizer = optim.Adam(model.parameters(), lr=self.bot.learning_rate)

        for agent in self.agents:
            # Training loop
            for episode in range(self.bot.num_episodes):
                # Initialize game
                hidden = torch.zeros(1, 1, self.bot.hidden_size)  # Initialize hidden state
                log_probs = []
                rewards = []
                model_actions = []
                agent_actions = []

                # Play one episode (multiple rounds)
                for _ in range(10):  # Play 10 rounds per episode
                    # Get current state (previous actions)
                    if len(log_probs) > 0:
                        state = torch.tensor([[self_action, opponent_action]], dtype=torch.float32)
                    else:
                        state = torch.tensor([[0, 0]], dtype=torch.float32)  # Initial state (cooperate to start)

                    # Get action probabilities
                    action_probs, hidden = model(state, hidden)
                    action_dist = torch.distributions.Categorical(action_probs)
                    action = action_dist.sample()
                    log_prob = action_dist.log_prob(action)
                    log_probs.append(log_prob)

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