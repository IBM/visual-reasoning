# Copyright (C) IBM Corporation 2020
#
# SPDX-License-Identifier: Apache-2.0

__author__ = "T.S. Jayram"

import torch
from torch import nn
from .utils import linear


class ReasoningUnit(nn.Module):
    """
    ``Reasoning Unit``
    """

    def __init__(self, dim, num_temporal):
        """
        Constructor for ``ReasoningUnit``

        Args:
            dim (int): common dimension of all objects
            num_temporal (int): number of temporal classes
        """

        # call base constructor
        super(ReasoningUnit, self).__init__()

        # 1 each for summarized measures of visual attention and read head
        in_dim = 2
        in_dim += num_temporal

        hidden_dim = 2 * in_dim

        # 1 dimension each for the following predicates:
        # image match, memory match, do replace, do add new
        out_dim = 4

        self.reasoning_module = torch.nn.Sequential(
            linear(in_dim, hidden_dim, bias=True),
            torch.nn.ELU(),
            linear(hidden_dim, hidden_dim, bias=True),
            torch.nn.ELU(),
            linear(hidden_dim, out_dim, bias=True),
            torch.nn.Sigmoid())

    def forward(self, control_state, visual_attention, read_head,
                temporal_weights):
        """Forward pass of the ``ReasoningUnit``

        Args:
            control_state (Tensor): [batch_size x dim]
            visual_attention (Tensor): [batch_size x (H*W)]
            read_head (Tensor): [batch_size x mem_size]
            temporal_weights (Tensor): [batch_size x num_temporal]

        Returns:
            Tensor: [batch_size]
                Image match soft predicate
            Tensor: [batch_size]
                Memory match soft predicate
            Tensor: [batch_size]
                Do replace soft predicate
            Tensor: [batch_size]
                Do add new soft predicate
        """

        # Compute a summary of each attention vector in [0,1]
        # The more the attention is localized, the more the
        # summary will be closer to 1

        va_aggregate = (visual_attention * visual_attention).sum(
            dim=-1, keepdim=True)
        rh_aggregate = (read_head * read_head).sum(dim=-1, keepdim=True)

        r_in = torch.cat(
            [temporal_weights, va_aggregate, rh_aggregate], dim=-1)
        r_out = self.reasoning_module(r_in)

        image_match = r_out[..., 0]
        memory_match = r_out[..., 1]
        do_replace = r_out[..., 2]
        do_add_new = r_out[..., 3]

        return image_match, memory_match, do_replace, do_add_new
