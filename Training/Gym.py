import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List
from Training.Agent import Agent
from Training.Bot import Bot
from torch.optim.lr_scheduler import StepLR


class Gym():
    def __init__(self, agents: List[Agent], bot: Bot):
        self.agents = agents
        self.bot = bot

    def RNN_train(self):
        model = self.bot.model
        optimizer = optim.Adam(model.parameters(), lr=self.bot.learning_rate)

        epsilon = self.bot.epsilon

        for episode in range(self.bot.num_episodes):
            for agent in self.agents:
                #self.bot.hidden = torch.zeros(1, 1, self.bot.hidden_size)  # Reset hidden state
                if self.bot.model_type == "LSTM":
                    self.bot.hidden = (
                        torch.zeros(1, 1, self.bot.hidden_size),  # h0
                        torch.zeros(1, 1, self.bot.hidden_size)   # c0
                    )
                else:
                    self.bot.hidden = torch.zeros(1, 1, self.bot.hidden_size)  # Reset hidden state

                log_probs = []
                rewards = []
                model_actions = []
                agent_actions = []
                self_action = 1  # Initial action (cooperate)
                opponent_action = agent.next_move(agent_moves=agent_actions, opponent_moves=model_actions)

                for _ in range(50):  # Play 50 rounds per episode
                    if len(log_probs) > 0:
                        self.bot.state = torch.tensor([[self_action.item(), opponent_action.item()]], dtype=torch.float32)
                        opponent_action = agent.next_move(agent_moves=agent_actions, opponent_moves=model_actions)
                    else:
                        self.bot.state = torch.tensor([[1, opponent_action]], dtype=torch.float32)  # Initial state

                    action_probs, self.bot.hidden = model(self.bot.state.unsqueeze(0), self.bot.hidden)
                    action_dist = torch.distributions.Categorical(action_probs)

                    # Epsilon-greedy exploration
                    if np.random.rand() < epsilon:
                        action = torch.tensor(np.random.choice([0, 1]), dtype=torch.long)
                    else:
                        action = action_dist.sample()

                    log_prob = action_dist.log_prob(action)
                    log_probs.append(log_prob)

                    model_actions.append(action)
                    agent_actions.append(opponent_action)

                    # Determine reward
                    if action.item() == 1 and opponent_action.item() == 1:
                        reward = torch.tensor([3])  # Mutual cooperation
                    elif action.item() == 1 and opponent_action.item() == 0:
                        reward = torch.tensor([0])  # Sucker's payoff
                    elif action.item() == 0 and opponent_action.item() == 1:
                        reward = torch.tensor([5])  # Temptation to defect
                    else:
                        reward = torch.tensor([1])  # Mutual defection
                    rewards.append(reward)

                    self_action = action
                
                gamma = 0.9  # Discount factor

                # Calculate discounted returns for each step
                discounted_returns = []
                G = 0
                for reward in reversed(rewards):
                    G = reward + gamma * G
                    discounted_returns.insert(0, G)

                discounted_returns = torch.tensor(discounted_returns, dtype=torch.float32)
                baseline = discounted_returns.mean()  # Baseline is the mean of returns

                optimizer.zero_grad()
                # Calculate policy loss using discounted returns minus baseline
                policy_loss = -sum([log_prob * (return_t - baseline) for log_prob, return_t in zip(log_probs, discounted_returns)])
                policy_loss.backward()
                optimizer.step()

                if episode % 20 == 0:
                    print(f"""Episode {episode}, Cumulative Reward: {int(sum(rewards))}, Avg Model Action: {round(float(sum(model_actions)/len(model_actions)), 2)}, Avg Agent Action: {round(float(sum(agent_actions)/len(agent_actions)), 2)}""")
                    # print(f"Model Actions: {[action.item() for action in model_actions]}")
                    # print(f"Agent Actions: {[action.item() for action in agent_actions]}")
                    #print(f"Rewards: {rewards}")
        self.bot.save_trained_state()  # Save the model state after training so it can be reset after each game

        