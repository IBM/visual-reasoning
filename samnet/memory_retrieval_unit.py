# Copyright (C) IBM Corporation 2020
#
# SPDX-License-Identifier: Apache-2.0

__author__ = "T.S. Jayram"

from torch.nn import Module

from .attention_module import AttentionModule
from .interaction_module import InteractionModule


class MemoryRetrievalUnit(Module):
    """
    ``Memory Retrieval Unit``: Retrieve a candidate object from memory
    """

    def __init__(self, dim):
        """
        Constructor for ``MemoryRetrievalUnit``

        Args:
            dim (int): dimension of feature objects
        """

        # call base constructor
        super(MemoryRetrievalUnit, self).__init__()

        # instantiate interaction module
        self.interaction_module = InteractionModule(dim)

        # instantiate attention module
        self.attention_module = AttentionModule(dim)

    def forward(self, summary_object, memory, control_state):
        """
        Forward pass of the ``MemoryRetrievalUnit``

        Args:
            summary_object (Tensor): [batch_size x dim]
                Previous summary object
            memory (Tensor): [batch_size x mem_size x dim]
            control_state (Tensor): [batch_size x dim]
                Current control state

        Returns:
            Tensor: [batch_size x dim] Visual_object
            Tensor: [batch_size x (H*W)] Visual attention vector
        """

        # Combine the summary object with VWM
        vwm_modified = self.interaction_module(summary_object, memory)

        # compute attention weights
        memory_object, read_head = self.attention_module(
            control_state, vwm_modified, memory)

        return memory_object, read_head
