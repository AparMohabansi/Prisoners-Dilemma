from typing import List, Tuple, Dict, Literal
from Training.Bot import Bot
from Training.Agent import Agent
from Training.Agents import *
# from Training.Agents.HumanPlayer import HumanPlayer
from Training.Gym import Gym
from Evaluation.Game import Game

# The user class, this is used to represent the user and its actions
class User:
    def __init__(self, name: str):
        self.name = name
        self.score = 0

    def choose_action(self):
        # Placeholder for decision logic
        action = int(input("Cooperate or Defect? (1 or 0) "))
        while action != 1 and action != 0:
            print(f"Incorrect input. {action}")
            action = input("Cooperate or Defect? (1 or 0)  ")
   
        return action

    def update_score(self, points: int):
        self.score += points

    

class GameOld:
    def __init__(self, player1, player2, rounds=5):
        self.player1 = player1
        self.player2 = player2
        self.rounds = rounds

    def play_round(self):
        player1_action = self.player1.next_move() 
        player2_action = self.player2.choose_action()

        if player1_action == 1:
            if player2_action == 1:
                print("Both players cooperated")
            else:
                print(f"Bot cooperated and {self.player2.name} defected")
        else:
            if player2_action == 1:
                print(f"Bot defected and {self.player2.name} cooperated")
            else:
                print("Both players defected")

    def play_game(self):
        for _ in range(self.rounds):
            self.play_round()

def main():
    prisoner1 = Bot("Prisoner 1")

    agents = [TitforTat()]
    gym = Gym(agents, prisoner1)
    gym.RNN_train()

    game = Game(HumanPlayer(), prisoner1, rounds=20)
    game.play_game()
    game.print_scores()

if __name__ == "__main__":
    main()