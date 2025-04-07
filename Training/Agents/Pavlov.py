from ..Agent import Agent
import torch

class Pavlov:
    def __init__(self):
        self.name = "Pavlov"
    
    def next_move(self, agent_moves, opponent_moves):
        if len(agent_moves) == 0 or len(opponent_moves) == 0:
            return torch.tensor(1)  # Start with cooperation
        
        # Win-Stay, Lose-Shift logic:
        # "Win" is mutual cooperation (1,1) or exploitation (0,1)
        # "Lose" is being exploited (1,0) or mutual defection (0,0)
        last_agent_move = agent_moves[-1]
        last_opponent_move = opponent_moves[-1]
        
        if last_agent_move == last_opponent_move:  # Both cooperated or both defected
            return last_agent_move.clone().detach()  # Repeat last move
        else:
            return (1 - last_agent_move).clone().detach()  # Change strategy