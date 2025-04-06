from Training.Bot import Bot
from Evaluation.Game import Game
from Training.Gym import Gym
from Training.Agents import TitforTat, AlwaysDefect, AlwaysCooperate, GrimTrigger, Random, Opposite
import random
from tqdm import tqdm

class Factory:
    def __init__(self):
        self.bots = []
        
    def create_bots(self, num_bots, 
                   hidden_size_range=(8, 32),
                   learning_rate_range=(0.005, 0.05),
                   num_episodes_range=(20, 50),
                   epsilon_range=(0.05, 0.2),
                   learning_rate_online_range=(0.05, 0.2),
                   memory_size_range=(10, 20),
                   entropy_coef_range=(0.001, 0.01),
                   model_type="LSTM",
                   verbose=False):
        """
        Create bots with randomized parameters within specified ranges
        
        Args:
            num_bots: Number of bots to create
            hidden_size_range: Range for hidden layer size (min, max)
            learning_rate_range: Range for learning rate (min, max)
            num_episodes_range: Range for number of training episodes (min, max)
            epsilon_range: Range for epsilon exploration parameter (min, max)
            learning_rate_online_range: Range for online learning rate (min, max)
            memory_size_range: Range for memory size (min, max)
            entropy_coef_range: Range for entropy coefficient (min, max)
            model_type: Type of model to use ("LSTM" or "GRU")
            verbose: Whether to print detailed training information
        
        Returns:
            List of trained bots
        """
        self.bots = []
        
        # Create progress bar for bot creation and training
        print(f"Creating and training {num_bots} bots with randomized parameters...")
        progress_bar = tqdm(total=num_bots, desc="Bot Training Progress")
        
        for i in range(num_bots):
            # Create bot with randomized name
            bot_name = f"Bot_{i+1}"
            bot = Bot(bot_name, output=False, verbose=verbose, model_type=model_type)
            
            # Set randomized parameters
            bot.hidden_size = random.randint(hidden_size_range[0], hidden_size_range[1])
            bot.learning_rate = random.uniform(learning_rate_range[0], learning_rate_range[1])
            bot.num_episodes = random.randint(num_episodes_range[0], num_episodes_range[1])
            bot.epsilon = random.uniform(epsilon_range[0], epsilon_range[1])
            bot.learning_rate_online = random.uniform(learning_rate_online_range[0], learning_rate_online_range[1])
            bot.memory_size = random.randint(memory_size_range[0], memory_size_range[1])
            bot.entropy_coef = random.uniform(entropy_coef_range[0], entropy_coef_range[1])
            
            # Create training agents
            agents = [
                TitforTat(),
                AlwaysDefect(),
                AlwaysCooperate(),
                GrimTrigger(),
                Opposite(),
                Random()
            ]
            
            # Train bot using Gym
            gym = Gym(agents, bot)
            gym.RNN_train()
            
            # Add trained bot to list
            self.bots.append(bot)
            
            # Log bot parameters if verbose
            if verbose:
                print(f"\nBot {i+1} Parameters:")
                print(f"  Hidden Size: {bot.hidden_size}")
                print(f"  Learning Rate: {bot.learning_rate:.6f}")
                print(f"  Episodes: {bot.num_episodes}")
                print(f"  Epsilon: {bot.epsilon:.6f}")
                print(f"  Online Learning Rate: {bot.learning_rate_online:.6f}")
                print(f"  Memory Size: {bot.memory_size}")
                print(f"  Entropy Coefficient: {bot.entropy_coef:.6f}")
            
            # Update progress bar
            progress_bar.update(1)
        
        progress_bar.close()
        print(f"Created and trained {num_bots} bots successfully!")
        return self.bots
    
    def get_bots(self):
        """Return the list of trained bots"""
        return self.bots
    
    def get_bot_params(self):
        """Return a list of parameter dictionaries for all bots"""
        bot_params = []
        for bot in self.bots:
            params = {
                "name": bot.name,
                "hidden_size": bot.hidden_size,
                "learning_rate": bot.learning_rate,
                "num_episodes": bot.num_episodes,
                "epsilon": bot.epsilon,
                "learning_rate_online": bot.learning_rate_online,
                "memory_size": bot.memory_size,
                "entropy_coef": bot.entropy_coef,
                "model_type": bot.model_type
            }
            bot_params.append(params)
        return bot_params