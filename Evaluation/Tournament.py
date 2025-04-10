from Training.Bot import Bot
from Evaluation.Game import Game
from typing import List
import random
from tqdm import tqdm

class Tournament:
    def __init__(self, bots: List[Bot], num_rounds: int = 5, num_games: int = 1):
        self.bots = bots
        self.num_rounds = num_rounds  # Number of rounds in each game
        self.num_games = num_games  # Each bot plays this many games
        self.scores = {bot: 0 for bot in bots}

    def play_match(self, bot1: Bot, bot2: Bot):
        game = Game(bot1, bot2, rounds=self.num_rounds)
        game.play_game()
        self.scores[bot1] += game.score[0]
        self.scores[bot2] += game.score[1]
        bot1.reset_to_trained_state()
        bot2.reset_to_trained_state()

    def run_tournament(self):
        # Initialize counters to track games played by each agent
        games_played = {bot: 0 for bot in self.bots}
        
        # Calculate total number of matches to play
        total_matches = (len(self.bots) * self.num_games) // 2
        
        # Create progress bar
        progress_bar = tqdm(
            total=total_matches,
            desc="Tournament Progress",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
        )
        matches_played = 0

        # Reset all bots to their trained state and enable online learning
        for bot in self.bots:
            bot.reset_to_trained_state()
            bot.online_learning = True  # Enable online learning for all bots
        
        # Continue until all bots have played their quota of games
        while matches_played < total_matches:
            # Create list of eligible bots (those who haven't played all their games)
            eligible_bots = [bot for bot in self.bots if games_played[bot] < self.num_games]
            
            # If only one bot is left needing games, we can't form any more pairs
            if len(eligible_bots) < 2:
                print(f"Warning: Cannot create more valid pairs. Some bots played {games_played} games.")
                break
            
            # Shuffle the eligible bots to randomize pairings
            random.shuffle(eligible_bots)
            
            # Create pairs from the shuffled eligible bots
            for i in range(0, len(eligible_bots) - 1, 2):
                if i + 1 < len(eligible_bots):
                    bot1 = eligible_bots[i]
                    bot2 = eligible_bots[i + 1]
                    
                    # Check if both bots still need games
                    if games_played[bot1] < self.num_games and games_played[bot2] < self.num_games:
                        # Play the match
                        self.play_match(bot1, bot2)
                        
                        # Update counters
                        games_played[bot1] += 1
                        games_played[bot2] += 1
                        matches_played += 1

                        # Update progress bar
                        progress_bar.update(1)
                        
                        # Break if we've reached the total number of matches
                        if matches_played >= total_matches:
                            break
        
        # Close the progress bar
        progress_bar.close()
        
        # Print summary
        print("\nTournament completed!")
        print(f"Matches played: {matches_played}")
        print(f"Rounds played per Game: {self.num_rounds}")

    def get_winner(self):
        return max(self.scores.values())
    
    def get_winners(self, num_winners: int = 1, print_winners: bool = True, 
                include_hyperparams: bool = False, output_file: str = None):
        """
        Get the top winners of the tournament with option to write to a file.
        
        Args:
            num_winners: Number of top winners to return
            print_winners: Whether to print the winners to the console
            include_hyperparams: Whether to print/output hyperparameters of winning bots
            output_file: Path to file where results should be written (optional)
        """
        num_winners = min(num_winners, len(self.scores))
        winners = sorted(self.scores.items(), key=lambda item: item[1], reverse=True)[:num_winners]
        
        # Create output content
        output_lines = ["Tournament winners:"]
        for bot, score in winners:
            output_lines.append(f"{bot.name} Score: {score}")
            
            # Add hyperparameters if requested
            if include_hyperparams:
                hyperparameters = bot.get_model_hyperparameters()
                output_lines.append(f"{bot.name} Hyperparameters:")
                for key, value in hyperparameters.items():
                    if key != "Name":
                        output_lines.append(f"{key}: {value}")
                output_lines.append(f"Number of Training Agents: {len(bot.training_agents)}")
                output_lines.append("-" * 30)
        
        # Print to console if requested
        if print_winners:
            for line in output_lines:
                print(line)
        
        # Write to file if path provided
        if output_file:
            with open(output_file, 'w') as f:
                for line in output_lines:
                    f.write(f"{line}\n")
        
        # Return values if not printing
        if not print_winners:
            return [score for _, score in winners]