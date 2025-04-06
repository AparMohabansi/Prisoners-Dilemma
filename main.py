from typing import List, Tuple, Dict, Literal
from Training.Bot import Bot
from Training.Agent import Agent
from Training.Agents import *
from Training.Gym import Gym
from Evaluation.Game import Game


def main():
    # Create the bot
    bot = Bot("Learning Bot", output=True, verbose=True)
    bot.hidden_size = 16
    bot.learning_rate = 0.03  # Increased learning rate for faster adaptation
    bot.num_episodes = 200  # Pre-training episodes
    
    # Create pre-training agents - use a variety for better generalization
    agents = [
        TitforTat()
        # AlwaysDefect(),  # Important to learn against defectors
        # AlwaysCooperate(),
        # GrimTrigger(),  # Teaches consequences of defection
        # Random()  # Adds unpredictability to training
    ]
    
    # Pre-train using the Gym
    print("Pre-training bot in Gym...")
    gym = Gym(agents, bot)
    gym.RNN_train()
    print("Pre-training complete!")
    
    # Create a human player
    human_player = HumanPlayer()
    
    # Enable online learning for the bot with faster adaptation
    bot.online_learning = True
    
    # Create the game with the pre-trained bot
    print("\nStarting game with pre-trained bot (online learning enabled)...")
    game = Game(human_player, bot, rounds=50)
    
    # Play the game - bot will continue learning from human interactions
    game.play_game()
    game.print_scores()


if __name__ == "__main__":
    main()