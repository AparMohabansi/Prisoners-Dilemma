from Agent import Agent
from typing import override

class AlwaysCooperate(Agent):
    @override
    def next_move(self, agent_moves, opponent_moves):
        return 1