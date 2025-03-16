from Agent import Agent
from typing import override
import random


class Random(Agent):
    @override
    def next_move(self, agent_moves, opponent_moves):
        return random.choice([0, 1])