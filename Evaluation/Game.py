from Training.config import SCORE_GUIDE
from Training.Agent import Agent
from Training.Bot import Bot
from typing import List, Tuple, Union

class Game():
    def __init__(self, player1: Union[Agent, Bot], player2: Union[Agent, Bot], rounds: int = 5):
        self.player1 = player1
        self.player2 = player2
        self.rounds = rounds
        self.score = [0, 0]
        self.player1_moves = []
        self.player2_moves = []

    def play_round(self):
        # Get each player's actions
        actions = (int(self.player1.next_move(agent_moves=self.player1_moves, opponent_moves=self.player2_moves)), 
                   int(self.player2.next_move(agent_moves=self.player2_moves, opponent_moves=self.player1_moves)))
        
        # Save the actions
        self.player1_moves.append(actions[0])
        self.player2_moves.append(actions[1])

        # Add the round's scores
        self.add_score(SCORE_GUIDE[actions])

    def play_game(self):
        for _ in range(self.rounds):
            self.play_round()
    
    def add_score(self, score: Tuple[int]) -> None:
        self.score[0] += score[0]
        self.score[1] += score[1]

    def print_scores(self) -> None:
        print(f"Player 1 Score: {self.score[0]}")
        print(f"Player 2 Score: {self.score[1]}")