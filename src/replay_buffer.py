from collections import deque, namedtuple
import numpy as np
import random
import torch


class ReplayBuffer:
    """A first-in-first-out buffer to store experiences."""

    
    def __init__(self, buffer_size):
        """Initializes the buffer.

        Params
        ======
            buffer_size (int): the maximum size of the buffer
        """
        self.buffer = deque(maxlen=buffer_size)  
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

        
    def add(self, state, action, reward, next_state, done):
        """Adds a new experience to the buffer.
        
        Params
        ======
            state (array_like): the current state
            action (int): the action taken
            reward (float): the reward received for taking the action in the state
            next_state (array_like): the next state
            done (bool): indicates whether the episode is done or not        
        """
        e = self.experience(state, action, reward, next_state, done)
        self.buffer.append(e)

        
    def sample(self, size):
        """Randomly samples experiences from the buffer (with replacement).
        
        Params
        ======
            size (int): the size of the sample
            
        Returns
        =======
            A tuple of vectors of the components of the experiences.
        """
        experiences = random.sample(self.buffer, k=size)
                
        states = torch.stack([e.state for e in experiences if e is not None], dim=0)
        actions = torch.stack([e.action for e in experiences if e is not None], dim=0)
        rewards = torch.stack([e.reward for e in experiences if e is not None], dim=0)
        next_states = torch.stack([e.next_state for e in experiences if e is not None], dim=0)
        dones = torch.stack([e.done for e in experiences if e is not None], dim=0)#.astype(np.uint8)
        
        return (states, actions, rewards, next_states, dones)


    def __len__(self):
        """Returns the current size of internal buffer."""
        return len(self.buffer)
