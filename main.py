from typing import List, Tuple, Dict, Literal
from Training.Bot import Bot
from Training.Agent import Agent
from Training.Agents import *
# from Training.Agents.HumanPlayer import HumanPlayer
from Training.Gym import Gym
from Evaluation.Game import Game


def main():
    prisoner1 = Bot("Prisoner 1")

    agents = [TitforTat()]
    gym = Gym(agents, prisoner1)
    gym.RNN_train()

    game = Game(HumanPlayer(), prisoner1, rounds=50)
    game.play_game()
    game.print_scores()

if __name__ == "__main__":
    main()