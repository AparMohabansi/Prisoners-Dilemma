from ..Agent import Agent
import torch

# MajorityRule - Does what the opponent has done most often
class MajorityRule:
    def __init__(self):
        self.name = "MajorityRule"
    
    def next_move(self, agent_moves, opponent_moves):
        if len(opponent_moves) == 0:
            return torch.tensor(1)  # Start with cooperation
        
        # Count cooperations and defections
        cooperations = sum(1 for move in opponent_moves if move == 1)
        defections = len(opponent_moves) - cooperations
        
        if cooperations > defections:
            return torch.tensor(1)  # Cooperate if opponent mostly cooperated
        elif defections > cooperations:
            return torch.tensor(0)  # Defect if opponent mostly defected
        else:
            return torch.tensor(1)  # Cooperate on ties