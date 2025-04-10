# Set OpenMP environment variable to avoid library loading conflicts
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from typing import List, Tuple, Dict, Literal
from Training.Bot import Bot
from Training.Agent import Agent
from Training.Agents import *
from Training.Gym import Gym
from Evaluation.Game import Game
from Evaluation.Tournament import Tournament
from Evaluation.Factory import Factory


def main():
    print_debug_output = False
    number_of_bots = 6
    number_of_rounds = 40
    number_of_games = 10
    output_file = "Tournament_Results.txt"
    stats_output_dir = os.path.join(os.getcwd(), "tournament_stats")

    # Create output directory if it doesn't exist
    if not os.path.exists(stats_output_dir):
        os.makedirs(stats_output_dir)

    # Create and train bots
    factory = Factory(output=print_debug_output)
    bots = factory.create_bots(num_bots=number_of_bots)

    # Run tournament
    tournament = Tournament(bots=bots, num_rounds=number_of_rounds, num_games=number_of_games)
    tournament.run_tournament()
    
    # Get and display winners
    tournament.get_winners(20, print_winners=True, include_hyperparams=True, output_file=output_file)
    
    # Export tournament statistics
    print("\nExporting tournament statistics...")
    csv_path, plots_dir = tournament.export_tournament_stats(stats_output_dir)
    print(f"Statistics exported to CSV: {csv_path}")
    print(f"Analysis plots saved to: {plots_dir}")

def verse_human():
    # Create the bot
    bot = Bot("Learning Bot", output=True, verbose=True, model_type="RNN")
    bot.hidden_size = 16
    bot.learning_rate = 0.03  # Increased learning rate for faster adaptation
    bot.num_episodes = 20  # Pre-training episodes
    
    # Create pre-training agents - use a variety for better generalization
    agents = [
        #TitforTat()
        #AlwaysDefect(),  # Important to learn against defectors
        #AlwaysCooperate() #,
        #GrimTrigger()  # Teaches consequences of defection
        #Opposite()
        Random()  # Adds unpredictability to training
    ]
    
    # Pre-train using the Gym
    print("Pre-training bot in Gym...")
    gym = Gym(agents, bot, output=True)
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