import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    """An actor network class. Each agent will have its own actor network, 
    that receives information only about that agent."""


    def __init__(self, state_shape_single, action_shape_single, hidden_size=256):
        """Builds the network and initializes its parameters.

        Params
        ======
            state_shape_single (int): The dimension of the per-agent state-space.
            action_shape_single (int): The dimension of the per-agent action-space.
            hidden_size (int): The number of nodes in the hidden layers.
        """
        super(Actor, self).__init__()

        self.layer_1 = nn.Linear(in_features=state_shape_single, out_features=hidden_size)
        self.layer_2 = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.layer_3 = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.layer_4 = nn.Linear(in_features=hidden_size, out_features=action_shape_single)


    def forward(self, state):
        """Passes the state of an agent through the network.

        Params
        ======
            state (array_like): The agent's state.

        Returns
        =======
            The result of the state passed through the network.
        """
        x = F.relu(self.layer_1(state))
        x = F.relu(self.layer_2(x))
        x = F.relu(self.layer_3(x))
        x = F.tanh(self.layer_4(x))  # So that the actions lie in the interval [-1, 1].

        return x


class Critic(nn.Module):
    """A critic network class. Each agent will have its own critic network, but each critic network
    receives the states and actions of all agents.
    """

    def __init__(self, state_shape_all, action_shape_all, hidden_size=256):
        """Builds the network and initializes its parameters.

        Params
        ======
            state_shape_all (tuple): The agent x state-space size.
            action_shape_all (tuple): The agent x action-space size.
            hidden_size (int): The number of nodes in the hidden layers.
        """
        super(Critic, self).__init__()

        state_shape_all_flat = np.prod(state_shape_all)
        action_shape_all_flat = np.prod(action_shape_all)

        self.layer_1 = nn.Linear(in_features=state_shape_all_flat, out_features=hidden_size // 4)
        self.layer_2 = nn.Linear(in_features=hidden_size // 4 + action_shape_all_flat, out_features=hidden_size)
        self.layer_3 = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.layer_4 = nn.Linear(in_features=hidden_size, out_features=1)


    def forward(self, states, actions):
        """Passes the states and actions of all agents through the network.

        Params
        ======
            states (array_like): The states of the agents.
            actions (array_like): The actions of the agents.

        Returns
        =======
            The result of the states and actions passed through the network.
        """

        states_flat = states.view(states.shape[0], -1)
        actions_flat = actions.view(actions.shape[0], -1)

        x = F.relu(self.layer_1(states_flat))
        x = torch.cat([x, actions_flat], dim=1)
        x = F.relu(self.layer_2(x))
        x = F.relu(self.layer_3(x))
        x = self.layer_4(x)

        return torch.squeeze(x, dim=1)
