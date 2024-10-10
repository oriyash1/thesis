class BattleOfSexesGame:
    reward_matrix = {
        'AA': (3, 2),  # Player 1 prefers Event A more
        'BB': (2, 3),  # Player 2 prefers Event B more
        'AB': (0, 0),  # They attend different events
        'BA': (0, 0)   # They attend different events
    }

    def __init__(self, agent1, agent2, reward_matrix, n_episodes, discount_factor):
        self.agent1 = agent1
        self.agent2 = agent2
        self.reward_matrix = reward_matrix
        self.n_episodes = n_episodes
        self.discount_factor = discount_factor
        self.history_agent1 = []  # Separate history for agent 1
        self.history_agent2 = []  # Separate history for agent 2 (even if it's a strategy, history might be useful)

    def compute_rewards(self, action1, action2):
        """Compute the rewards for both agents based on their actions."""
        if action1 == 0 and action2 == 0:  # Both choose Event A
            return self.reward_matrix['AA']
        elif action1 == 1 and action2 == 1:  # Both choose Event B
            return self.reward_matrix['BB']
        elif action1 == 0 and action2 == 1:  # Player 1 chooses A, Player 2 chooses B
            return self.reward_matrix['AB']
        elif action1 == 1 and action2 == 0:  # Player 1 chooses B, Player 2 chooses A
            return self.reward_matrix['BA']

    def play_round(self):
        # Agent 1 chooses its action based on its own history
        action1 = self.agent1.q_learning_select_action(self.history_agent1)

        # Check if agent2 is a callable strategy function or an LSTM agent
        if callable(self.agent2):
            action2 = self.agent2(self.history_agent2)  # If agent2 is a strategy (e.g., tit-for-tat)
        else:
            action2 = self.agent2.q_learning_select_action(self.history_agent2)  # If agent2 is a learning agent

        # Record both actions in their respective histories
        self.history_agent1.append((action1, action2))
        self.history_agent2.append((action1, action2))

        # Compute rewards based on the actions
        reward1, reward2 = self.compute_rewards(action1, action2)

        # Update each agent with the reward and the history
        self.agent1.update_strategy(reward1, (action1, action2), self.history_agent1)
        if not callable(self.agent2):  # Only update agent2 if it's an LSTM agent
            self.agent2.update_strategy(reward2, (action1, action2), self.history_agent2)

        # Print the actions of both agents
        action_names = ["Event A", "Event B"]
        print(
            f"Episode {len(self.history_agent1)}: Agent 1 chose {action_names[action1]}, Agent 2 chose {action_names[action2]}")

        return reward1, reward2

    def play_game(self):
        """Play the full game across all episodes, returning the cumulative rewards for both agents."""
        rewards_agent1 = []
        rewards_agent2 = []
        total_rewards_agent1 = []
        total_rewards_agent2 = []

        # Play each episode and store the immediate rewards
        for episode in range(self.n_episodes):
            reward1, reward2 = self.play_round()
            rewards_agent1.append(reward1)
            rewards_agent2.append(reward2)

        # Now calculate the cumulative rewards for each episode starting from the current episode
        for episode in range(self.n_episodes):
            episode_reward1 = 0
            episode_reward2 = 0
            cumulative_discount = 1  # Reset the discount for each episode

            # Calculate the cumulative reward for each episode from the current one to the end
            for future_episode in range(episode, self.n_episodes):
                episode_reward1 += rewards_agent1[future_episode] * cumulative_discount
                episode_reward2 += rewards_agent2[future_episode] * cumulative_discount
                cumulative_discount *= self.discount_factor  # Update the discount factor

            # Print the cumulative reward for each episode
            print(f"Episode {episode + 1}: Total Reward 1: {episode_reward1}, Total Reward 2: {episode_reward2}")

            # Store the cumulative reward for each episode in the lists
            total_rewards_agent1.append(episode_reward1)
            total_rewards_agent2.append(episode_reward2)

        # Return the lists of cumulative rewards for both agents
        return total_rewards_agent1, total_rewards_agent2
