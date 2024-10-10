import numpy as np
# from game import IteratedPrisonersDilemma
#  from game2 import BattleOfSexesGame
from game3 import MatchingPennies


class Training:
    def __init__(self, agent1, agent2, n_games, n_episodes_per_game, reward_matrix, discount_factor):
        """
        Initialize the training process with two agents.

        :param agent1: The first agent participating in the training (LSTMAgent).
        :param agent2: The second agent participating in the training (LSTMAgent or any other agent or strategy function).
        :param n_games: The number of games to play during training.
        :param n_episodes_per_game: The number of episodes per game.
        :param reward_matrix: The payoff matrix, typically T > R > P > S.
        :param discount_factor: Discount rate for the rewards (γ).
        """
        self.agent1 = agent1
        self.agent2 = agent2
        self.n_games = n_games
        self.n_episodes_per_game = n_episodes_per_game
        self.rewards_agent1 = []
        self.rewards_agent2 = []
        self.game = MatchingPennies(agent1, agent2, reward_matrix, n_episodes_per_game, discount_factor)

    def train(self):
        """
        Train the agents over the specified number of games.
        """
        for game_index in range(self.n_games):
            # Reset history or any stateful information between games if required
            self.agent1.history = []
            if hasattr(self.agent2, 'history'):
                self.agent2.history = []

            total_reward1, total_reward2 = self.game.play_game()

            # Log the total rewards for analysis
            self.rewards_agent1.append(total_reward1)
            self.rewards_agent2.append(total_reward2)

            if (game_index + 1) % 100 == 0:
                print(
                    f"Game {game_index + 1}/{self.n_games}: Agent 1 Total Reward = {total_reward1}, Agent 2 Total Reward = {total_reward2}")

        self.analyze_rewards()

        # After training, print Q-values for each agent if applicable
        print("Q-values for Agent 1:")
        self.agent1.print_q_values()
        if hasattr(self.agent2, 'print_q_values'):
            print("Q-values for Agent 2:")
            self.agent2.print_q_values()

    def analyze_rewards(self):
        """
        Analyze the cumulative rewards collected during training across all episodes,
        and calculate the average reward for each agent.
        """
        avg_reward_agent1 = np.mean(self.rewards_agent1)
        avg_reward_agent2 = np.mean(self.rewards_agent2)

        # Print out the average rewards for each agent
        print(f"Average Reward for Agent 1 across all episodes: {avg_reward_agent1}")
        print(f"Average Reward for Agent 2 across all episodes: {avg_reward_agent2}")

