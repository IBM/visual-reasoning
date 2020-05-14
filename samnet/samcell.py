# Copyright (C) IBM Corporation 2020
#
# SPDX-License-Identifier: Apache-2.0

"""
samcell.py: Implementation of the SAMCell for the SAMNet model.
"""
__author__ = "T.S. Jayram"

from torch import nn

from .visual_retrieval_unit import VisualRetrievalUnit
from .summary_unit import SummaryUpdateUnit
from .memory_retrieval_unit import MemoryRetrievalUnit
from .reasoning_unit import ReasoningUnit


class SAMCell(nn.Module):
    """
    ``SAMCell``: executes a single reasoning step of a recurent process
    """

    def __init__(self, dim, num_temporal):
        """
        Constructor for ``SAMCell``
        Args:
            dim (int): common dimension of all objects
            num_temporal (int): number of temporal classes
        """

        super(SAMCell, self).__init__()

        # instantiate all the units within VWM_cell
        self.visual_retrieval_unit = VisualRetrievalUnit(dim)
        self.memory_retrieval_unit = MemoryRetrievalUnit(dim)
        self.reasoning_unit = ReasoningUnit(dim, num_temporal)
        self.summary_unit = SummaryUpdateUnit(dim)

    def forward(self, summary_object, control_all,
                feature_map, feature_map_proj, memory):
        """Forward pass of ``SAMCell``

        Args:
            summary_object (Tensor): [batch_size x dim]
            control_all (tuple): info from the question driven controller
                control state, control attention, temporal_weights
            feature_map (Tensor): [batch_size x (H*W) x dim]
                Frame representation
            feature_map_proj (Tensor): [batch_size x (H*W) x dim]
                A pre-computed projection of the feature map
            memory (Tensor): [batch_size x mem_size x dim]

        Returns:
            Tensor: [batch_size x dim]
                New summary object
            tuple: (visual_object, read_head, do_replace, do_add_new)
        """

        control_state, control_attention, temporal_weights = control_all

        # visual retrieval unit, obtain visual output and visual attention
        visual_object, visual_attention = self.visual_retrieval_unit(
            summary_object, feature_map, feature_map_proj, control_state)

        # memory retrieval unit, obtain memory output and memory attention
        memory_object, read_head = self.memory_retrieval_unit(
            summary_object, memory, control_state)

        # reason about the objects
        image_match, memory_match, do_replace, do_add_new = (
            self.reasoning_unit(
                control_state, visual_attention, read_head, temporal_weights))

        # summary update Unit
        new_summary_object = self.summary_unit(
                summary_object, image_match, visual_object,
                memory_match, memory_object)

        # package all the SAM Cell state info
        sam_cell_info = dict(
            vo=visual_object, rhd=read_head, do_r=do_replace, do_a=do_add_new)

        return new_summary_object, sam_cell_info
