from ..Agent import Agent
import torch

class TitForTwoTats:
    def __init__(self):
        self.name = "TitForTwoTats"
    
    def next_move(self, agent_moves, opponent_moves):
        if len(opponent_moves) >= 2 and opponent_moves[-1] == 0 and opponent_moves[-2] == 0:
            return torch.tensor(0)  # Defect if opponent defected twice
        return torch.tensor(1)