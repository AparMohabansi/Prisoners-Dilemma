from ..Agent import Agent
import random
import torch


class Random(Agent):
    def next_move(self, agent_moves, opponent_moves):
        return random.choice([torch.tensor([0]), torch.tensor([1])])