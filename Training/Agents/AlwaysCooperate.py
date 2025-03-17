from ..Agent import Agent
import torch

class AlwaysCooperate(Agent):
    def next_move(self, agent_moves, opponent_moves):
        return torch.tensor([1])