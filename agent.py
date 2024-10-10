import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


def init_weights(m):
    """
    Custom weight initialization for the network layers.
    """
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.1)


class LSTMAgent(nn.Module):  # Inherit from nn.Module
    def __init__(self, learning_rate, discount_factor, epsilon, epsilon_decay, n_memory,
                 lstm_hidden_size=64, lstm_layers=1):
        super(LSTMAgent, self).__init__()  # Initialize nn.Module
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.n_memory = n_memory
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_layers = lstm_layers

        # Define the LSTM-based network architecture for Q-learning
        self.lstm = nn.LSTM(input_size=2, hidden_size=self.lstm_hidden_size, num_layers=self.lstm_layers,
                            batch_first=True)
        self.fc1 = nn.Linear(self.lstm_hidden_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 2)  # Output Q-values for each action (Cooperate, Defect)

        # Initialize weights
        self.apply(init_weights)

        # Optimizer and loss function
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()

        # Initialize history and Q-values log
        self.history = []
        self.q_values_log = []

    def forward(self, state):
        """
        Perform a forward pass through the network.

        :param state: The input state sequence (batch_size, sequence_length, input_size).
        :return: Q-values for the actions.
        """
        # Pass the input through the LSTM layer
        lstm_out, _ = self.lstm(state)

        # Ensure lstm_out has the expected shape: (batch_size, sequence_length, lstm_hidden_size)
        if len(lstm_out.shape) == 2:
            # If the output has only 2 dimensions, reshape it to add the sequence length dimension
            lstm_out = lstm_out.unsqueeze(0)

        # Select the output from the last time step
        x = lstm_out[:, -1, :]  # Shape: (batch_size, lstm_hidden_size)

        # Pass through the fully connected layers
        x = self.relu(self.fc1(x))
        q_values = self.fc2(x)

        return q_values

    def q_learning_select_action(self, history):
        state = self.get_state_from_history(history)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)  # Convert state to tensor and add batch dimension

        random_value = np.random.rand()  # Generate random number once

        if random_value < self.epsilon:
            action = np.random.choice([0, 1])  # Exploration
            print('epsilon', self.epsilon)
            print('random', random_value)
        else:
            with torch.no_grad():
                q_values = self.forward(state_tensor)
            action = torch.argmax(q_values).item()

        self.epsilon = max(self.epsilon * self.epsilon_decay, 0.01)  # Ensure epsilon decays
        return action

    def update_strategy(self, reward, current_state, history):
        """
        Update the Q-values using the Q-learning update rule.
        The agent updates the Q-value of the action it took in the previous state
        based on the reward received and the maximum expected future reward.

        :param reward: The reward received after taking an action.
        :param current_state: The current state after the action was taken.
        :param history: The entire history of the game.
        """
        previous_state = self.get_state_from_history(history[:-1])
        action = history[-1][0]  # The action taken by the agent in the previous state.

        # Convert the state history to tensors.
        previous_state_tensor = torch.FloatTensor(previous_state).unsqueeze(0)
        current_state_tensor = torch.FloatTensor(current_state).unsqueeze(0)

        with torch.no_grad():
            # Calculate the maximum Q-value for the current state, which represents the future reward.
            future_q_value = torch.max(self.forward(current_state_tensor)).item()

            # Compute the target Q-value based on the reward and discounted future reward.
            target_q_value = reward + self.discount_factor * future_q_value
            # Ensure the target Q-value is non-negative
            target_q_value = max(target_q_value, 0)

        # Get the current Q-value for the action taken in the previous state.
        q_values = self.forward(previous_state_tensor)
        current_q_value = q_values[0, action]

        updated_q_value = (1 - self.learning_rate) * current_q_value + self.learning_rate * target_q_value

        q_values[0, action] = updated_q_value
        # Convert the target Q-value to a tensor
        target_q_value_tensor = torch.tensor([target_q_value], dtype=torch.float32)

        # Ensure both current Q-value and target Q-value are tensors, and compute the loss.
        loss = self.loss_fn(current_q_value.unsqueeze(0), target_q_value_tensor)

        # Perform backpropagation to update the model weights.
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Store the Q-values for later analysis, ensuring no negative values.
        q_values = torch.clamp(q_values, min=0.0)
        self.q_values_log.append(q_values.detach().numpy())

    # def update_strategy(self, reward, current_state, history):
    #     """
    #     Update the Q-values using the Q-learning update rule.
    #     The agent updates the Q-value of the action it took in the previous state
    #     based on the reward received and the maximum expected future reward.
    #
    #     :param reward: The reward received after taking an action.
    #     :param current_state: The current state after the action was taken.
    #     :param history: The entire history of the game.
    #     """
    #     previous_state = self.get_state_from_history(history[:-1])
    #     action = history[-1][0]  # The action taken by the agent in the previous state.
    #
    #     # Convert the state history to tensors.
    #     previous_state_tensor = torch.FloatTensor(previous_state).unsqueeze(0)
    #     current_state_tensor = torch.FloatTensor(current_state).unsqueeze(0)
    #
    #     with torch.no_grad():
    #         # Calculate the maximum Q-value for the current state, which represents the future reward.
    #         future_q_value = torch.max(self.forward(current_state_tensor)).item()
    #
    #         # Compute the target Q-value based on the reward and discounted future reward.
    #         target_q_value = reward + self.discount_factor * future_q_value
    #         target_q_value = max(target_q_value, 0)  # Ensure the target Q-value is non-negative
    #
    #     # Get the current Q-value for the action taken in the previous state.
    #     q_values = self.forward(previous_state_tensor)
    #     current_q_value = q_values[0, action]
    #
    #     # Compute the updated Q-value using the learning rate
    #     updated_q_value = (1 - self.learning_rate) * current_q_value + self.learning_rate * target_q_value
    #
    #     # Replace the current Q-value with the updated one
    #     q_values[0, action] = updated_q_value
    #
    #     # Convert the target Q-value to a tensor (for loss calculation)
    #     target_q_value_tensor = torch.tensor([target_q_value], dtype=torch.float32)
    #
    #     # Ensure both current Q-value and target Q-value are tensors, and compute the loss.
    #     loss = self.loss_fn(current_q_value.unsqueeze(0), target_q_value_tensor)
    #
    #     # Perform backpropagation to update the model weights.
    #     self.optimizer.zero_grad()
    #     loss.backward()
    #     self.optimizer.step()
    #
    #     # Store the Q-values for later analysis.
    #     self.q_values_log.append(q_values.detach().numpy())

    # def update_strategy(self, reward, current_state, history):
    #     """
    #     Update the Q-values using the Q-learning update rule.
    #     The agent updates the Q-value of the action it took in the previous state
    #     based on the reward received and the maximum expected future reward.
    #
    #     :param reward: The reward received after taking an action.
    #     :param current_state: The current state after the action was taken.
    #     :param history: The entire history of the game.
    #     """
    #     previous_state = self.get_state_from_history(history[:-1])
    #     action = history[-1][0]  # The action taken by the agent in the previous state.
    #
    #     # Convert the state history to tensors.
    #     previous_state_tensor = torch.FloatTensor(previous_state).unsqueeze(0)
    #     current_state_tensor = torch.FloatTensor(current_state).unsqueeze(0)
    #
    #     with torch.no_grad():
    #         # Calculate the maximum Q-value for the current state, which represents the future reward.
    #         future_q_value = torch.max(self.forward(current_state_tensor)).item()
    #
    #         # Compute the target Q-value based on the reward and discounted future reward.
    #         target_q_value = reward + self.discount_factor * future_q_value
    #         target_q_value = max(target_q_value, 0)  # Ensure the target Q-value is non-negative
    #
    #     # Get the current Q-value for the action taken in the previous state.
    #     q_values = self.forward(previous_state_tensor)
    #     current_q_value = q_values[0, action]
    #
    #     # Update the Q-value using the learning rate
    #     updated_q_value = (1 - self.learning_rate) * current_q_value + self.learning_rate * target_q_value
    #
    #     # Replace the current Q-value with the updated one
    #     q_values[0, action] = updated_q_value
    #
    #     # Perform backpropagation to update the model weights (if you are using a neural network)
    #     target_q_value_tensor = torch.tensor([target_q_value], dtype=torch.float32)
    #     loss = self.loss_fn(current_q_value.unsqueeze(0), target_q_value_tensor)
    #     self.optimizer.zero_grad()
    #     loss.backward()
    #     self.optimizer.step()
    #
    #     # Store the Q-values for later analysis, ensuring no negative values.
    #     q_values = torch.clamp(q_values, min=0.0)
    #     self.q_values_log.append(q_values.detach().numpy())

        # Debugging: Print the Q-values to monitor learning
        # print(f"Q-values for previous state: {q_values}")
        # print(f"Target Q-value: {target_q_value}, Updated Q-value: {updated_q_value}")

    def get_state_from_history(self, history):
        if len(history) < self.n_memory:
            padded_history = [(0, 0)] * (self.n_memory - len(history)) + history
        else:
            padded_history = history[-self.n_memory:]

        return padded_history

    def print_q_values(self):
        """
        Print the stored Q-values for analysis.
        """
        for i, q_vals in enumerate(self.q_values_log):
            print(f"Episode {i + 1}: Q-values = {q_vals}")
