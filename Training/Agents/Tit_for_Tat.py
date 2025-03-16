from Agent import Agent
from typing import override

class TitforTat(Agent):
    @override
    def next_move(self, agent_moves, opponent_moves):
        return opponent_moves[-1] if len(opponent_moves) > 0 else 0