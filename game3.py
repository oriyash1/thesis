class MatchingPennies:
    reward_matrix = {
        'match': 1,  # Reward for Player 1 if actions match
        'mismatch': -1  # Reward for Player 1 if actions don't match (Player 2 wins)
    }

    def __init__(self, agent1, agent2, reward_matrix, n_episodes, discount_factor):
        """
        Initialize the game with two agents, a reward matrix, and the number of episodes.

        :param agent1: The first agent participating in the game (can be a learning agent).
        :param agent2: The second agent participating in the game (can be a learning agent).
        :param n_episodes: Number of episodes the game will be played.
        :param discount_factor: Discount factor for cumulative reward (γ).
        """
        self.agent1 = agent1
        self.agent2 = agent2
        self.n_episodes = n_episodes
        self.reward_matrix = reward_matrix
        self.discount_factor = discount_factor
        self.history = []

    def compute_rewards(self, action1, action2):
        """
        Compute the rewards for both agents based on their actions.

        :param action1: Action taken by agent1 (0 = Heads, 1 = Tails).
        :param action2: Action taken by agent2 (0 = Heads, 1 = Tails).
        :return: Tuple of rewards for both agents.
        """
        if action1 == action2:  # Both actions match (Player 1 wins)
            return self.reward_matrix['match'], -self.reward_matrix['match']
        else:  # Actions mismatch (Player 2 wins)
            return self.reward_matrix['mismatch'], -self.reward_matrix['mismatch']

    def play_round(self):
        """
        Play a single round of the game, where both agents choose an action and receive rewards.

        :return: Tuple of rewards for both agents.
        """
        # Get the actions of both agents
        action1 = self.agent1.q_learning_select_action(self.history)

        # If agent2 is a callable function (random or fixed strategy), call it directly; otherwise, use q_learning_select_action
        if callable(self.agent2):
            action2 = self.agent2(self.history)
        else:
            action2 = self.agent2.q_learning_select_action(self.history)

        # Record the actions in the history
        self.history.append((action1, action2))

        # Compute rewards based on the payoff matrix
        reward1, reward2 = self.compute_rewards(action1, action2)

        # Update agents with the reward and resulting new state
        self.agent1.update_strategy(reward1, self.history[-1], self.history)
        if not callable(self.agent2):  # Only update strategy if agent2 is a learning agent
            self.agent2.update_strategy(reward2, self.history[-1], self.history)

        # Print the actions of both agents
        action_names = ["Heads", "Tails"]
        print(
            f"Episode {len(self.history)}: Agent 1 chose {action_names[action1]}, Agent 2 chose {action_names[action2]}")

        return reward1, reward2

    def play_game(self):
        """
        Play the full game across all episodes, returning the cumulative rewards for both agents.

        :return: Lists of cumulative rewards for agent1 and agent2 across all episodes.
        """
        rewards_agent1 = []
        rewards_agent2 = []
        total_rewards_agent1 = []
        total_rewards_agent2 = []

        # Play each episode and store the immediate rewards
        for episode in range(self.n_episodes):
            reward1, reward2 = self.play_round()
            rewards_agent1.append(reward1)
            rewards_agent2.append(reward2)

        # Now calculate the cumulative rewards for each episode
        for episode in range(self.n_episodes):
            episode_reward1 = 0
            episode_reward2 = 0
            cumulative_discount = 1  # Reset the discount factor for each episode

            # Calculate the cumulative reward from the current episode to the last one
            for future_episode in range(episode, self.n_episodes):
                episode_reward1 += rewards_agent1[future_episode] * cumulative_discount
                episode_reward2 += rewards_agent2[future_episode] * cumulative_discount
                cumulative_discount *= self.discount_factor  # Update the discount factor

            # Print cumulative rewards for each episode
            print(f"Episode {episode + 1}: Total Reward 1: {episode_reward1}, Total Reward 2: {episode_reward2}")

            # Store cumulative rewards in the lists
            total_rewards_agent1.append(episode_reward1)
            total_rewards_agent2.append(episode_reward2)

        # Return the lists of cumulative rewards for both agents
        return total_rewards_agent1, total_rewards_agent2
