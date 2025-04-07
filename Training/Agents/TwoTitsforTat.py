from ..Agent import Agent
import torch

class TwoTitsForTat:
    def __init__(self):
        self.name = "TwoTitsForTat"
        self.defect_counter = 0
    
    def next_move(self, agent_moves, opponent_moves):
        if len(opponent_moves) == 0:
            return torch.tensor(1)  # Start with cooperation
        
        if opponent_moves[-1] == 0:  # If opponent just defected
            self.defect_counter = 2  # Set counter to defect twice
        
        if self.defect_counter > 0:
            self.defect_counter -= 1
            return torch.tensor(0)  # Defect
        
        return torch.tensor(1)  # Cooperate