from ..Agent import Agent
import torch

class Opposite(Agent):
    def next_move(self, agent_moves, opponent_moves):
        if len(opponent_moves)>0 and opponent_moves[-1] == 0:
            return torch.tensor([1])
        else:
            return torch.tensor([0])