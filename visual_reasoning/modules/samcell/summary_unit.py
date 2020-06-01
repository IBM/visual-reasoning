# Copyright (C) IBM Corporation 2020
#
# SPDX-License-Identifier: Apache-2.0

__author__ = "T.S. Jayram"

import torch
from torch import nn
from .utils import Linear


class SummaryUpdateUnit(nn.Module):
    """
    ``Summary Update Unit``
    """

    def __init__(self, dim):
        """Constructor for ``SummaryUpdateUnit``

        Args:
            dim (int): common dimension of all objects
        """

        # call base constructor
        super(SummaryUpdateUnit, self).__init__()

        # linear layer for projecting context_output and summary_output
        self.concat_layer = Linear(2 * dim, dim, bias=True)

    def forward(self, summary_object, image_match, visual_object,
                memory_match, memory_object):
        """

        Args:
            summary_object (Tensor): [batch_size x dim]
            image_match (Tensor): [batch_size]
            visual_object (Tensor): [batch_size x dim]
            memory_match (Tensor): [batch_size]
            memory_object (Tensor): [batch_size x dim]

        Returns:
            Tensor: [batch_size x dim]
                New summary object
        """

        # compute new relevant object
        relevant_object = (image_match[..., None] * visual_object
                           + memory_match[..., None] * memory_object)

        # combine the new read vector with the prior memory state (w1)
        new_summary_object = self.concat_layer(
            torch.cat([relevant_object, summary_object], dim=1))

        return new_summary_object
