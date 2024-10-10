from evaluation import Evaluation
from utils import save_model, load_model
from agent import LSTMAgent
from training import Training


def tit_for_tat(history):
    if len(history) == 0:
        return 0  # Cooperate on the first move
    else:
        return history[-1][0]  # Mirror the opponent's last move


def always_cooperate(history):
    return 0  # Always cooperate


def always_defect(history):
    return 1  # Always defect


def main():
    # Parameters from the article
    n_games = 1 # Number of games
    n_episodes_per_game = 10  # Number of episodes per game
    learning_rate = 0.01  # Learning rate
    discount_factor = 0.95  # Discount factor
    epsilon = 0.2  # Epsilon for exploration
    epsilon_decay = 0.9999  # Epsilon decay rate
    n_memory = 5  # Number of previous episodes remembered by the LSTM

    # Define the reward matrix
    # reward_matrix = {
    #     'R': 3,  # Reward for mutual cooperation
    #     'S': 0,  # Sucker's payoff (Cooperate, Defect)
    #     'T': 5,  # Temptation to defect (Defect, Cooperate)
    #     'P': 1   # Punishment for mutual defection
    # }
    # reward_matrix = {
    #     'AA': (3, 2),  # Player 1 prefers Event A more
    #     'BB': (2, 3),  # Player 2 prefers Event B more
    #     'AB': (0, 0),  # They attend different events
    #     'BA': (0, 0)  # They attend different events
    # }
    reward_matrix = {
        'match': 1,  # Reward for Player 1 if actions match
        'mismatch': -1  # Reward for Player 1 if actions don't match (Player 2 wins)
    }

    # Define your agents with the specified parameters
    agent1 = LSTMAgent(
        learning_rate=learning_rate,
        discount_factor=discount_factor,
        epsilon=epsilon,
        epsilon_decay=epsilon_decay,
        n_memory=n_memory
    )
    agent2 = always_defect

    # Train the agents
    training_instance = Training(
        agent1,
        agent2,
        n_games=n_games,
        n_episodes_per_game=n_episodes_per_game,
        reward_matrix=reward_matrix,
        discount_factor=discount_factor
    )
    training_instance.train()

    # Evaluate the agents
    evaluator = Evaluation(agent1, agent2, training_instance)
    # evaluator.compare_agents()
    # evaluator.analyze_behavior()
    # evaluator.plot_rewards()

    # Save the trained models
    save_model(agent1, "agent1_model.pth")
    # Note: Since agent2 is a strategy function, not a model, you don't need to save it with save_model


if __name__ == "__main__":
    main()

