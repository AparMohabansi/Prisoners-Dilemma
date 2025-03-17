from ..Agent import Agent
import torch

class GrimTrigger(Agent):
    def __init__(self):
        self.hasDefected = False

    def next_move(self, agent_moves, opponent_moves):
        if len(opponent_moves)>0 and opponent_moves[-1] == torch.tensor([0]):
            self.hasDefected = True
        
        if self.hasDefected:
            return torch.tensor([0])
        else:
            return torch.tensor([1])