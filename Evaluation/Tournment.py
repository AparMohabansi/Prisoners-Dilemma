from Training.Agent import Agent
from Training.Bot import Bot
from Training.config import SCORE_GUIDE
from typing import List

class Tournament:
    def __init__(self, agents: List[Agent], bots: List[Bot], num_rounds: int = 5, num_games: int = 1):
        self.agents = agents
        self.bots = bots
        self.num_rounds = num_rounds
        self.num_games = num_games
        self.scores = {agent: 0 for agent in agents + bots}
        self.noise = 0.0

    def play_match(self, agent1: Agent, agent2: Agent):
        for _ in range(self.num_rounds):
            move1 = agent1.next_move([], [])
            move2 = agent2.next_move([], [])
            score1, score2 = self.get_scores(move1, move2)
            self.scores[agent1] += score1
            self.scores[agent2] += score2

    def get_scores(self, move1: int, move2: int):
        return SCORE_GUIDE[(move1, move2)]

    def run_tournament(self):
        for _ in range(self.num_games):
            for i in range(len(self.agents)):
                for j in range(i + 1, len(self.agents)):
                    self.play_match(self.agents[i], self.agents[j])
                for bot in self.bots:
                    self.play_match(self.agents[i], bot)

    def get_winner(self):
        return max(self.scores, key=self.scores.get)

    def set_rounds(self, num_rounds: int):
        self.num_rounds = num_rounds

    def set_noise(self, noise: float):
        self.noise = noise

# Example usage
if __name__ == "__main__":
    from Training.Agent import Agent


    class RandomAgent(Agent):
        def next_move(self, agent_moves, opponent_moves):
            import random
            return random.choice([0, 1])

    agents = [RandomAgent() for _ in range(4)]
    bots = [RandomAgent() for _ in range(2)]
    tournament = Tournament(agents, bots)
    tournament.set_rounds(10)
    tournament.set_noise(0.1)
    tournament.run_tournament()
    winner = tournament.get_winner()
    print(f"The winner is: {winner}")