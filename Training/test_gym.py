import torch
import numpy as np
from Gym import Gym
from Bot import Bot, RNNAgent
from Agent import Agent

# Mock Agent class for testing
class MockAgent(Agent):
    def __init__(self, name):
        super().__init__(name)
        self.score = 0

    def choose_action(self):
        # Randomly choose to cooperate (1) or defect (0)
        return np.random.randint(0, 2)

    def update_score(self, points):
        self.score += points

# Test function for the Gym class
def test_gym():
    # Hyperparameters
    input_size = 2  # [self_action, opponent_action]
    hidden_size = 16
    output_size = 2  # [Cooperate, Defect]
    learning_rate = 0.01
    num_episodes = 1000

    # Initialize bot and agents
    bot = Bot()
    agents = [MockAgent("Agent1"), MockAgent("Agent2")]

    # Initialize Gym
    gym = Gym(agents, bot)

    # Test RNN_train method
    print("Testing RNN_train method...")
    gym.RNN_train()

    # Test playing a round with the trained bot
    print("\nTesting play_round method with trained bot...")
    for _ in range(10):  # Play 10 rounds
        state = torch.tensor([[0, 0]], dtype=torch.float32)  # Initial state
        bot_action = bot.predict(state)
        print(f"Bot's action: {'Cooperate' if bot_action == 1 else 'Defect'}")

        # Simulate opponent's action (random for testing)
        opponent_action = np.random.randint(0, 2)
        print(f"Opponent's action: {'Cooperate' if opponent_action == 1 else 'Defect'}")

        # Determine reward
        if bot_action == 1 and opponent_action == 1:
            reward = 3  # Mutual cooperation
        elif bot_action == 1 and opponent_action == 0:
            reward = 0  # Sucker's payoff
        elif bot_action == 0 and opponent_action == 1:
            reward = 5  # Temptation to defect
        else:
            reward = 1  # Mutual defection
        print(f"Reward: {reward}\n")

# Run the test
if __name__ == "__main__":
    test_gym()