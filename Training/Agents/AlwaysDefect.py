from Agent import Agent
from typing import override

class AlwaysDefect(Agent):
    @override
    def next_move(self, agent_moves, opponent_moves):
        return 0