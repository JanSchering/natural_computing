from typing import List
import random

import torch as t
from torch.utils.checkpoint import checkpoint


class NCA(t.nn.Module):

    def __init__(self, update_net: t.nn.Module, min_steps:int, max_steps:int, p_update=0.5):
        super().__init__()
        self.min_steps = min_steps
        self.max_steps = max_steps
        self.device = "cuda" if t.cuda.is_available() else "cpu"
        self.update_net = update_net
        self.p_update = p_update

    def step(self, state:t.Tensor, rand_update_mask:t.Tensor) -> t.Tensor:
        """
        Simulate a step of the NCA from the given state, using the update mask.
        Returns the next state of the NCA.
        """
        update = self.update_net(state)
        state = (state + update * rand_update_mask)
        return state

    def forward(self, state:t.Tensor) -> List[t.Tensor]:
        """
        Forward a state through the NCA - Perform a sequence of [<min_steps>, <max_steps>] update steps of the given NCA state.
        Returns the sequence of updated states.
        """
        states = [state]

        for j in range(random.randint(self.min_steps, self.max_steps)):
            rand_update_mask = (t.rand((state.shape[0], 1, state.shape[2], state.shape[3]), device=self.device) < self.p_update).to(t.float32)
            state = checkpoint(self.step, state, rand_update_mask)
            states.append(state)

        return states
