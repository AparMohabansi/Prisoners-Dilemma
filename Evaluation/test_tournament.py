import sys
import os
# Add the project directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Training.Agent import Agent
from Evaluation.Tournment import Tournament

class RandomAgent(Agent):
    def next_move(self, agent_moves, opponent_moves):
        import random
        return random.choice([0, 1])

def main():
    agents = [RandomAgent() for _ in range(4)]
    bots = [RandomAgent() for _ in range(2)]
    tournament = Tournament(agents, bots)
    tournament.set_rounds(10)
    tournament.set_noise(0.1)
    tournament.run_tournament()
    winner = tournament.get_winner()
    print(f"The winner is: {winner}")

if __name__ == "__main__":
    main()