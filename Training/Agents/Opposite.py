from Agent import Agent
from typing import override

class Opposite(Agent):
    @override
    def next_move(self, agent_moves, opponent_moves):
        if opponent_moves[-1] == 0:
            return 1
        else:
            return 0