from Agent import Agent
from typing import override

class GrimTrigger(Agent):
    def __init__(self, id):
        self.hasDefected = False

    @override
    def next_move(self, agent_moves, opponent_moves):
        if opponent_moves[-1] == 0:
            self.hasDefected = True
        
        if self.hasDefected:
            return 0
        else:
            return 1