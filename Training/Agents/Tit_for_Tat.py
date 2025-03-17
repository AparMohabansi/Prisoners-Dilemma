from ..Agent import Agent
import torch

class TitforTat(Agent):
    def next_move(self, agent_moves, opponent_moves):
        if len(opponent_moves) > 0:
            return opponent_moves[-1]
        else:
            return torch.tensor([1])  # Cooperate on the first round