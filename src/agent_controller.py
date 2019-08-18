from networks import Actor, Critic
import numpy as np
import random
from replay_buffer import ReplayBuffer
import torch


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class AgentController():
    """An agent controller that interacts with and learns from its environment.
    Learning is done using the Multi-Agent Deep Deterministic Policy Gradient (MADDPG) algorithm 
    (see https://arxiv.org/abs/1706.02275).
    """

    def __init__(self,
                 agents_n,
                 state_shape_all,
                 action_shape_all,
                 buffer_size=int(1e6),
                 buffer_size_before_learning=500,
                 update_every=1,
                 epochs_n=3,
                 batch_size=256,
                 noise_weight_start=1,
                 noise_weight_factor=0.999,
                 noise_weight_min=0.1,
                 gamma=0.99,
                 tau=0.01,
                 actor_learning_rate=3e-4,
                 critic_learning_rate=3e-4,
                ):
        """Initializes an Agent Controller instance.
        
        Params
        ======
            agents_n (int): The number of agents.
            state_shape_all (tuple): The agent x state-space size.
            action_shape_all (tuple): The agent x action-space size.
            buffer_size (int): The replay buffer size.
            buffer_size_before_learning (int): The minimum buffer size before learning starts.
            update_every (int): Learning occurs each multiple of this number of steps.
            epochs_n (int): The number of times a minibatch is sampled at each learning step for each agent.
            batch_size (int): The minibatch size.
            noise_weight_start (float): The start weight applied to the standard Guassian noise that is added to actions.
            noise_weight_factor (float): The factor by which the noise weight is updated with after each noise step.
            noise_weight_factor_min (float): The minimum value to be used by the noise weight.
            gamma (float): The reward discount factor.
            tau (float): Used for soft-updating the parameters of the "slow" networks.
            actor_learning_rate (float): The learning rate used for the actor networks.
            critic_learning_rate (float): The learning rate used for the critic networks.
        """
        self.agents_n = agents_n
        self.state_shape_all = state_shape_all
        self.action_shape_all = action_shape_all
        self.buffer_size = buffer_size
        self.buffer_size_before_learning = buffer_size_before_learning
        self.update_every = update_every
        self.epochs_n = epochs_n
        self.batch_size = batch_size
        self.noise_weight_start = noise_weight_start
        self.noise_weight_factor = noise_weight_factor
        self.noise_weight_min = noise_weight_min
        self.gamma = gamma
        self.tau = tau
        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate

        #####
        
        assert self.agents_n == 2
        assert self.state_shape_all[0] == 2
        assert self.action_shape_all[0] == 2

        self.state_shape_single = self.state_shape_all[1]
        self.action_shape_single = self.action_shape_all[1]
        
        self.noise_weight = self.noise_weight_start
        
        # Create the networks:

        def create_actor():
            return Actor(self.state_shape_single, self.action_shape_single).to(device)

        def create_critic():
            return Critic(self.state_shape_all, self.action_shape_all).to(device)

        self.agent_networks = []
        for _ in range(agents_n):
            self.agent_networks.append(
                {'actor': {'fast': create_actor(), 'slow': create_actor()},
                 'critic': {'fast': create_critic(), 'slow': create_critic()}}
            )

        # Sync the initial weights between the "fast" and "slow" networks:
        for agent_network in self.agent_networks:
            self._perform_soft_update(agent_network['actor']['fast'], agent_network['actor']['slow'], tau=1)
            self._perform_soft_update(agent_network['critic']['fast'], agent_network['critic']['slow'], tau=1)            
                
        # Add optimizers:
        for agent_network in self.agent_networks:
            agent_network['actor']['optimizer'] = torch.optim.Adam(agent_network['actor']['fast'].parameters(), lr=self.actor_learning_rate)
            agent_network['critic']['optimizer'] = torch.optim.Adam(agent_network['critic']['fast'].parameters(), lr=self.critic_learning_rate)
            
        # Instantiate a replay buffer for memories:
        self.buffer = ReplayBuffer(self.buffer_size)        
        
        # Start counting the steps:
        self.t_step = 0


    def act(self, states):
        """Returns actions for each of the agents.

        Params
        ======
            states (array_like): The current states of the agents.
        """
        states = torch.from_numpy(states).float().to(device)
        actions = np.zeros(self.action_shape_all)
        for i in range(self.agents_n):
            actor_fast = self.agent_networks[i]['actor']['fast']
            actor_fast.eval()
            with torch.no_grad():
                actions[i,:] = actor_fast(states[i]).detach().cpu().numpy() + self.noise_weight * torch.randn(self.action_shape_all[1])  # Add Gaussian noise.
            actor_fast.train()
        return actions


    def step(self, states, actions, rewards, next_states, dones):
        """Adds the experience to the replay buffer. If it is an internal "update step", also
        learns for each agent.
        
        Params
        ======
            states (array_like): The current states for the agents.
            actions (array-like): The actions just taken by the agents.
            rewards (array-like): The rewards just received by the agents.
            next_states (array_like): The resulting states for the agents.
            dones (array-like): Indicates whether or not the episode is done for each agent.
        """

        experience = tuple(torch.from_numpy(component).float().cpu() for component in (states, actions, rewards, next_states, dones))
        self.buffer.add(*experience)

        # Possibly learn.
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            # Check whether there are enough experiences in the buffer.
            if len(self.buffer) > max(self.buffer_size_before_learning, self.batch_size):
                # Repeat the learning "self.epochs_n" times for each agent.
                for _ in range(self.epochs_n):
                    experiences = tuple(component.detach().to(device) for component in self.buffer.sample(self.batch_size))
                    for agent_i in range(self.agents_n):
                        self._learn(agent_i, experiences)
                    for component in experiences:
                        component.detach().cpu()
                # Update the noise weight.
                self.noise_weight = max(self.noise_weight_min, self.noise_weight * self.noise_weight_factor)


    def save(self, path):
        """Saves the weights of the various networks to a path."""
        
        for i in range(self.agents_n):
            torch.save(self.agent_networks[i]['actor']['fast'].state_dict(), path + f'_actor_{i}.pth')
            torch.save(self.agent_networks[i]['critic']['fast'].state_dict(), path + f'_critic_{i}.pth')


    def _learn(self, agent_i, experiences):
        """Uses the sampled experiences (from all agents at first) to train a specific agent.

        Params
        ======
            agent_i (int): The index of the agent to train.
            experiences (array-like): The experiences that form the minibatch.
        """
        states, actions, rewards, next_states, dones = experiences

        actor_fast = self.agent_networks[agent_i]['actor']['fast']
        actor_slow = self.agent_networks[agent_i]['actor']['slow']
        actor_optimizer = self.agent_networks[agent_i]['actor']['optimizer']
        critic_fast = self.agent_networks[agent_i]['critic']['fast']
        critic_slow = self.agent_networks[agent_i]['critic']['slow']
        critic_optimizer = self.agent_networks[agent_i]['critic']['optimizer']        
        
        # Update the Critic:
        actions_next_slow = torch.stack([self.agent_networks[k]['actor']['slow'](next_states[:,k,:]) for k in range(self.agents_n)], dim=1)        
        critic_predictions = critic_fast(states, actions)  # Note: All the states and all the actions.
        critic_targets = rewards[:,agent_i] + self.gamma * critic_slow(next_states, actions_next_slow) * (1 - dones[:,agent_i])                
        critic_errors = critic_predictions - critic_targets
        torch.Tensor.clamp_(critic_errors, min=-1, max=1)  # Note.
        critic_loss = torch.mean(torch.pow(critic_errors, 2))
        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()
        
        # Update the Actor:
        actions_current_fast = torch.stack([self.agent_networks[k]['actor']['fast'](states[:,k,:]) for k in range(self.agents_n)], dim=1)
        actor_loss = -torch.mean(critic_fast(states, actions_current_fast))  # Note: Want to maximise the expected return.
        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()

        # Update the slow networks:
        self._perform_soft_update(critic_fast, critic_slow, tau=self.tau)
        self._perform_soft_update(actor_fast, actor_slow, tau=self.tau)


    def _perform_soft_update(self, fast_network, slow_network, tau):
        """Performs a soft update of the slow network parameters.
        θ_slow = τ * θ_fast + (1 - τ) * θ_slow.

        Params
        ======
            fast_network: The network from which the weights are updated.
            slow_network: The network to which the weights are updated.
            tau (float): The interpolation parameter.
        """
        for slow_param, fast_param in zip(slow_network.parameters(), fast_network.parameters()):
            slow_param.data.copy_(tau * fast_param.data + (1.0 - tau) * slow_param.data)
