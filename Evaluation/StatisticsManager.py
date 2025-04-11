import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from Training.Bot import Bot
import pandas as pd

class StatisticsManager:
    """Utility class for analyzing and comparing statistics across multiple bots"""
    
    def __init__(self, output_dir=None):
        """Initialize the statistics manager"""
        if output_dir is None:
            self.output_dir = os.path.join(os.getcwd(), "stats_analysis")
        else:
            self.output_dir = output_dir
            
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
        self.bot_stats = {}
        
    def add_bot(self, bot: Bot):
        """Add a bot to track statistics for"""
        self.bot_stats[bot] = bot.get_statistics()
            
    def export_all_stats(self):
        """Export statistics for all bots to CSV file"""
        stats_data = []
        
        for bot, stats in self.bot_stats.items():
            # Flatten the stats dictionary
            flat_stats = {
                'name': bot.name,
                'model_type': stats['model_type'],
                'hidden_size': stats['hidden_size'],
                'learning_rate': bot.learning_rate,
                'num_episodes': bot.num_episodes,
                'epsilon': bot.epsilon,
                'online_learning_rate': bot.learning_rate_online,
                'memory_size': bot.memory_size,
                'entropy_coefficient': bot.entropy_coef,
                'training_agent_names': ','.join(bot.training_agents) if bot.training_agents else '',
                'num_training_agents': len(bot.training_agents) if bot.training_agents else 0,
                'cooperation_rate': stats['cooperation_rate'],
                'retaliation_rate': stats['retaliation_rate'],
                'forgiveness_rate': stats['forgiveness_rate'],
                'win_rate': stats['win_rate'],
                'avg_points_per_round': stats['avg_points_per_round'],
                'avg_score_per_game': stats['avg_score_per_game'],
                'score_variance': stats['score_variance'],
                'training_agents': stats['training_agents']
            }            
            stats_data.append(flat_stats)
            
        # Convert to DataFrame and export
        if stats_data:
            df = pd.DataFrame(stats_data)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(self.output_dir, f"all_bot_stats_{timestamp}.csv")
            df.to_csv(filepath, index=False)
            print(f"All bot statistics exported to {filepath}")
            return filepath
        else:
            print("No bot statistics to export")
            return None
            
    def plot_comparative_analysis(self):
        """Generate comparative plots across all bots"""
        if len(self.bot_stats) < 1:
            print("Not enough bots to create comparative analysis")
            return self.output_dir
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Extract data for plotting
        bot_names = [bot.name for bot in self.bot_stats.keys()]
        model_types = [stats['model_type'] for stats in self.bot_stats.values()]
        hidden_sizes = [stats['hidden_size'] for stats in self.bot_stats.values()]
        avg_scores = [stats['avg_score_per_game'] for stats in self.bot_stats.values()]
        avg_points_per_round = [stats['avg_points_per_round'] for stats in self.bot_stats.values()]
        cooperation_rates = [stats['cooperation_rate'] for stats in self.bot_stats.values()]
        retaliation_rates = [stats['retaliation_rate'] for stats in self.bot_stats.values()]
        forgiveness_rates = [stats['forgiveness_rate'] for stats in self.bot_stats.values()]
        win_rates = [stats['win_rate'] for stats in self.bot_stats.values()]
        score_variances = [stats['score_variance'] for stats in self.bot_stats.values()]
        
        # Additional metrics
        num_training_agents = [len(bot.training_agents) if bot.training_agents else 0 for bot in self.bot_stats.keys()]
        learning_rates = [bot.learning_rate for bot in self.bot_stats.keys()]
        epsilons = [bot.epsilon for bot in self.bot_stats.keys()]
        entropy_coefficients = [bot.entropy_coef for bot in self.bot_stats.keys()]
        memory_sizes = [bot.memory_size for bot in self.bot_stats.keys()]
        
        # 1. Strategy Components vs Score (3 side-by-side 2D plots)
        plt.figure(figsize=(15, 5))
        
        # Cooperation Rate vs Score
        plt.subplot(1, 3, 1)
        scatter1 = plt.scatter(cooperation_rates, avg_scores, 
                             c=win_rates, s=80, alpha=0.7, cmap='viridis')
        plt.colorbar(scatter1, label='Win Rate')
        plt.xlabel('Cooperation Rate')
        plt.ylabel('Average Score per Game')
        plt.title('Cooperation vs Performance')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Retaliation Rate vs Score
        plt.subplot(1, 3, 2)
        scatter2 = plt.scatter(retaliation_rates, avg_scores, 
                             c=win_rates, s=80, alpha=0.7, cmap='viridis')
        plt.colorbar(scatter2, label='Win Rate')
        plt.xlabel('Retaliation Rate')
        plt.ylabel('Average Score per Game')
        plt.title('Retaliation vs Performance')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Forgiveness Rate vs Score
        plt.subplot(1, 3, 3)
        scatter3 = plt.scatter(forgiveness_rates, avg_scores, 
                             c=win_rates, s=80, alpha=0.7, cmap='viridis')
        plt.colorbar(scatter3, label='Win Rate')
        plt.xlabel('Forgiveness Rate')
        plt.ylabel('Average Score per Game')
        plt.title('Forgiveness vs Performance')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        strategy_components_path = os.path.join(self.output_dir, f"strategy_components_{timestamp}.png")
        plt.savefig(strategy_components_path)
        plt.close()
        
        # 2. Strategy Balance Index vs Performance
        plt.figure(figsize=(10, 6))
        # Create a composite balance index (higher = more balanced strategy)
        balance_index = [(c + f - r)/3 for c, r, f in zip(cooperation_rates, retaliation_rates, forgiveness_rates)]
        
        scatter = plt.scatter(balance_index, avg_scores, c=[0 if t == "RNN" else 1 for t in model_types],
                            s=100, alpha=0.7, cmap='coolwarm')
        
        plt.colorbar(scatter, ticks=[0.25, 0.75], label='Model Type').set_ticklabels(['RNN', 'LSTM'])
        plt.xlabel('Strategy Balance Index ((cooperation + forgiveness - retaliation)/3)')
        plt.ylabel('Average Score per Game')
        plt.title('Strategy Balance vs Performance')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        balance_path = os.path.join(self.output_dir, f"strategy_balance_{timestamp}.png")
        plt.savefig(balance_path)
        plt.close()
        
        # 3. Training Methodology Impact
        plt.figure(figsize=(12, 6))
        
        # Left plot - Number of training agents vs Score
        plt.subplot(1, 2, 1)
        scatter1 = plt.scatter(num_training_agents, avg_scores, 
                              c=[0 if t == "RNN" else 1 for t in model_types],
                              s=80, alpha=0.7, cmap='coolwarm')
            
        plt.xlabel('Number of Training Agents')
        plt.ylabel('Average Score per Game')
        plt.title('Training Diversity vs Performance')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Right plot - Epsilon vs Score 
        plt.subplot(1, 2, 2)
        scatter2 = plt.scatter(epsilons, avg_scores, 
                              c=entropy_coefficients, 
                              s=80, alpha=0.7, cmap='plasma')
            
        plt.colorbar(scatter2, label='Entropy Coefficient')
        plt.xlabel('Exploration (Epsilon)')
        plt.ylabel('Average Score per Game')
        plt.title('Exploration vs Performance')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        train_method_path = os.path.join(self.output_dir, f"training_methodology_{timestamp}.png")
        plt.savefig(train_method_path)
        plt.close()
        
        # 4. Performance Consistency Plot
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(avg_scores, score_variances, 
                             c=win_rates, s=100, alpha=0.7, cmap='RdYlGn')
            
        plt.colorbar(scatter, label='Win Rate')
        plt.xlabel('Average Score per Game')
        plt.ylabel('Score Variance')
        plt.title('Performance vs Consistency')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        consistency_path = os.path.join(self.output_dir, f"performance_consistency_{timestamp}.png")
        plt.savefig(consistency_path)
        plt.close()
        
        # 5. Model Architecture and Memory Effects
        plt.figure(figsize=(12, 6))
        
        # Left plot - Hidden Size vs Score by Model Type
        plt.subplot(1, 2, 1)
        for model_type in set(model_types):
            indices = [i for i, mt in enumerate(model_types) if mt == model_type]
            plt.scatter([hidden_sizes[i] for i in indices], 
                       [avg_scores[i] for i in indices], 
                       label=model_type, alpha=0.7, s=80)
            
        plt.xlabel('Hidden Size')
        plt.ylabel('Average Score per Game')
        plt.title('Model Architecture Performance')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Right plot - Memory Size vs Score
        plt.subplot(1, 2, 2)
        scatter = plt.scatter(memory_sizes, avg_scores, 
                             c=learning_rates, s=80, alpha=0.7, cmap='viridis')
            
        plt.colorbar(scatter, label='Learning Rate')
        plt.xlabel('Memory Size')
        plt.ylabel('Average Score per Game')
        plt.title('Memory Size Performance')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        architecture_path = os.path.join(self.output_dir, f"model_architecture_{timestamp}.png")
        plt.savefig(architecture_path)
        plt.close()

        # 6. Training Agent Impact Analysis
        plt.figure(figsize=(14, 8))
        
        # List of all possible training agents
        agent_types = [
            "TitforTat",
            "AlwaysDefect",
            "AlwaysCooperate",
            "GrimTrigger",
            "Opposite",
            "Random",
            "TitForTwoTats",
            "TwoTitsForTat",
            "Pavlov",
            "MajorityRule"
        ]
        
        # Calculate average score for bots trained with each agent type
        agent_stats = []
        for agent_type in agent_types:
            # Find bots trained with this agent
            trained_indices = [i for i, bot in enumerate(self.bot_stats.keys()) 
                              if agent_type in bot.training_agents]
            not_trained_indices = [i for i, bot in enumerate(self.bot_stats.keys()) 
                                 if agent_type not in bot.training_agents]
            
            # Calculate average scores
            if trained_indices:
                avg_score_trained = np.mean([avg_scores[i] for i in trained_indices])
                count_trained = len(trained_indices)
            else:
                avg_score_trained = 0
                count_trained = 0
                
            if not_trained_indices:
                avg_score_not_trained = np.mean([avg_scores[i] for i in not_trained_indices])
                count_not_trained = len(not_trained_indices)
            else:
                avg_score_not_trained = 0
                count_not_trained = 0
            
            agent_stats.append({
                'agent': agent_type,
                'avg_score_trained': avg_score_trained,
                'avg_score_not_trained': avg_score_not_trained,
                'count_trained': count_trained,
                'count_not_trained': count_not_trained
            })
        
        # Sort by differential impact (how much does training with this agent help)
        for stats in agent_stats:
            if stats['count_trained'] > 0 and stats['count_not_trained'] > 0:
                stats['impact'] = stats['avg_score_trained'] - stats['avg_score_not_trained']
            elif stats['count_trained'] > 0:
                stats['impact'] = stats['avg_score_trained']
            else:
                stats['impact'] = 0
                
        agent_stats.sort(key=lambda x: x['impact'], reverse=True)
        
        # Plot bar chart
        agents = [stat['agent'] for stat in agent_stats]
        scores_trained = [stat['avg_score_trained'] for stat in agent_stats]
        scores_not_trained = [stat['avg_score_not_trained'] for stat in agent_stats]
        counts_trained = [stat['count_trained'] for stat in agent_stats]
        
        # Bar width and positions
        bar_width = 0.35
        x = np.arange(len(agents))
        
        # Create bars
        plt.bar(x - bar_width/2, scores_trained, bar_width, label='Trained with agent', 
                alpha=0.7, color='forestgreen')
        plt.bar(x + bar_width/2, scores_not_trained, bar_width, label='Not trained with agent', 
                alpha=0.7, color='firebrick')
        
        # Add count labels
        for i, count in enumerate(counts_trained):
            plt.text(i - bar_width/2, scores_trained[i] + max(avg_scores)*0.02, 
                    f"n={count}", ha='center', fontsize=9)
        for i, count in enumerate(agent_stats):
            if count['count_not_trained'] > 0:
                plt.text(i + bar_width/2, scores_not_trained[i] + max(avg_scores)*0.02, 
                        f"n={count['count_not_trained']}", ha='center', fontsize=9)
        
        # Customize plot
        plt.xlabel('Training Agent Type')
        plt.ylabel('Average Score per Game')
        plt.title('Impact of Training Agent Types on Performance')
        plt.xticks(x, agents, rotation=45, ha='right')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5, axis='y')
        
        # Add explanatory text
        plt.figtext(0.5, 0.01, 
                  "This chart compares the average performance of bots trained with each agent type vs. those not trained with it.\n"
                  "Agents are sorted by their positive impact on performance.",
                  ha='center', fontsize=9)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])  # Adjust layout to make room for text
        agent_impact_path = os.path.join(self.output_dir, f"agent_impact_{timestamp}.png")
        plt.savefig(agent_impact_path)
        plt.close()

        print(f"Comparative analysis plots saved to {self.output_dir}")
        return self.output_dir
