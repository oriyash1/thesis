import numpy as np
import matplotlib.pyplot as plt
from collections import Counter


class Evaluation:
    def __init__(self, agent1, agent2, training_instance):
        """
        Initialize the evaluation process with two agents and a training instance.

        :param agent1: The first agent (trained LSTMAgent).
        :param agent2: The second agent (trained LSTMAgent or any other agent).
        :param training_instance: The training instance containing rewards and training data.
        """
        self.agent1 = agent1
        self.agent2 = agent2
        self.training_instance = training_instance

    # def compare_agents(self):
    #     """
    #     Compare the performance of the two agents based on their cumulative rewards.
    #     """
    #     rewards1 = np.array(self.training_instance.rewards_agent1)
    #     rewards2 = np.array(self.training_instance.rewards_agent2)
    #
    #     avg_reward1 = np.mean(rewards1)
    #     avg_reward2 = np.mean(rewards2)
    #
    #     print(f"Agent 1 Average Reward: {avg_reward1}")
    #     print(f"Agent 2 Average Reward: {avg_reward2}")
    #
    #     return avg_reward1, avg_reward2

    def analyze_behavior(self):
        """
        Analyze the behavior of the agents by examining their cooperation/defection rates,
        transition probabilities, and Markov matrices.
        """
        action_history = self.training_instance.agent1.history
        total_episodes = len(action_history)

        if total_episodes == 0:
            print("No action history available for analysis.")
            return

        # Count cooperation and defection for each agent
        agent1_cooperate = sum(1 for a1, a2 in action_history if a1 == 0)
        agent1_defect = total_episodes - agent1_cooperate

        agent2_cooperate = sum(1 for a1, a2 in action_history if a2 == 0)
        agent2_defect = total_episodes - agent2_cooperate

        # Calculate cooperation/defection rates
        agent1_coop_rate = agent1_cooperate / total_episodes
        agent1_defect_rate = agent1_defect / total_episodes

        agent2_coop_rate = agent2_cooperate / total_episodes
        agent2_defect_rate = agent2_defect / total_episodes

        print(f"Agent 1 Cooperation Rate: {agent1_coop_rate:.2f}, Defection Rate: {agent1_defect_rate:.2f}")
        print(f"Agent 2 Cooperation Rate: {agent2_coop_rate:.2f}, Defection Rate: {agent2_defect_rate:.2f}")

        # Generate Markov matrices for each agent
        markov_matrix_agent1, markov_matrix_agent2 = self.generate_markov_matrices(action_history)

        print("Agent 1 Markov Matrix:")
        print(markov_matrix_agent1)
        print("Agent 2 Markov Matrix:")
        print(markov_matrix_agent2)

    def generate_markov_matrices(self, action_history):
        """
        Generate the Markov matrices for both agents based on the action history.

        :param action_history: A list of tuples containing the previous actions of both agents.
        :return: The Markov matrices for agent1 and agent2.
        """
        transition_counts_agent1 = Counter()
        transition_counts_agent2 = Counter()

        for i in range(1, len(action_history)):
            prev_actions = action_history[i - 1]
            current_actions = action_history[i]

            # Update transition counts for agent 1
            transition_counts_agent1[(prev_actions[0], current_actions[0])] += 1

            # Update transition counts for agent 2
            transition_counts_agent2[(prev_actions[1], current_actions[1])] += 1

        # Calculate transition probabilities for agent 1
        markov_matrix_agent1 = np.zeros((2, 2))
        for (prev_action, next_action), count in transition_counts_agent1.items():
            markov_matrix_agent1[prev_action, next_action] = count

        # Normalize to get probabilities
        markov_matrix_agent1 = markov_matrix_agent1 / markov_matrix_agent1.sum(axis=1, keepdims=True)

        # Calculate transition probabilities for agent 2
        markov_matrix_agent2 = np.zeros((2, 2))
        for (prev_action, next_action), count in transition_counts_agent2.items():
            markov_matrix_agent2[prev_action, next_action] = count

        # Normalize to get probabilities
        markov_matrix_agent2 = markov_matrix_agent2 / markov_matrix_agent2.sum(axis=1, keepdims=True)

        return markov_matrix_agent1, markov_matrix_agent2

    def plot_rewards(self):
        """
        Plot the cumulative rewards of the two agents over time.
        """
        plt.plot(self.training_instance.rewards_agent1, label="Agent 1 Rewards")
        plt.plot(self.training_instance.rewards_agent2, label="Agent 2 Rewards")
        plt.xlabel("Game Number")
        plt.ylabel("Cumulative Reward")
        plt.title("Cumulative Rewards Over Time")
        plt.legend()
        plt.show()


