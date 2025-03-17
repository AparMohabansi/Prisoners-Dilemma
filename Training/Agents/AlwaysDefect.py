from ..Agent import Agent
import torch

class AlwaysDefect(Agent):
    def next_move(self, agent_moves, opponent_moves):
        return torch.tensor([0])