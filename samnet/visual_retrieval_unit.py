# Copyright (C) IBM Corporation 2020
#
# SPDX-License-Identifier: Apache-2.0

__author__ = "T.S. Jayram"

from torch import nn
from .interaction_module import InteractionModule
from .attention_module import AttentionModule


class VisualRetrievalUnit(nn.Module):
    """
    ``Visual Retrieval Unit``: Retrieve a candidate object from memory
    """

    def __init__(self, dim):
        """
        Constructor for ``VisualRetrievalUnit``

        Args:
            dim (int): common dimension of all objects
        """

        # call base constructor
        super(VisualRetrievalUnit, self).__init__()

        # instantiate interaction module
        self.interaction_module = InteractionModule(dim, do_project=False)

        # instantiate attention module
        self.attention_module = AttentionModule(dim)

    def forward(self, summary_object, feature_map, feature_map_proj,
                control_state):
        """Forward pass of the ``VisualRetrievalUnit``

        Args:
            summary_object (Tensor): [batch_size x dim]
                Previous summary object
            feature_map (Tensor): [batch_size x (H*W) x dim]
                Frame representation
            feature_map_proj (Tensor): [batch_size x (H*W) x dim]
                A pre-computed projection of the feature map
            control_state (Tensor): [batch_size x dim]
                Current control state

        Returns:
            Tensor: [batch_size x dim]
                Visual_object
            Tensor: [batch_size x (H*W)]
                Visual attention vector
        """

        feature_map_modified = self.interaction_module(
            summary_object, feature_map, feature_map_proj)

        # compute attention weights
        visual_object, visual_attention = self.attention_module(
            control_state, feature_map_modified, feature_map)

        return visual_object, visual_attention
