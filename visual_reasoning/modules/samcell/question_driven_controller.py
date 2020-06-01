# Copyright (C) IBM Corporation 2020
#
# SPDX-License-Identifier: Apache-2.0

__author__ = "T.S. Jayram"

import torch
from torch import nn

from .utils import Linear
from .attention_module import AttentionModule


class QuestionDrivenController(nn.Module):
    """
    ``Question Driven Controller``
    """

    def __init__(self, dim, max_step, num_temporal_classes):
        """Constructor for QuestionDrivenController

        Args:
            dim (int): common dimension of all objects
            max_step (int): maximum number of reasoning steps
            num_temporal_classes (int): number of temporal classes
        """

        # call base constructor
        super(QuestionDrivenController, self).__init__()

        # define the linear layers (one per step) used to make the questions
        # encoding
        self.pos_aware_layers = torch.nn.ModuleList()
        for _ in range(max_step):
            self.pos_aware_layers.append(Linear(2 * dim, dim, bias=True))

        # define the linear layer used to create the cqi values
        self.ctrl_question = Linear(2 * dim, dim, bias=True)

        # instantiate attention module
        self.attention_module = AttentionModule(dim)

        # temporal classifier that outputs 4 classes
        self.temporal_classifier = torch.nn.Sequential(
            Linear(dim, dim, bias=True),
            torch.nn.ELU(),
            Linear(dim, num_temporal_classes, bias=True),
            torch.nn.Softmax(dim=-1)
            )

    def forward(self, step, contextual_words, question_encoding, ctrl_state):
        """Forward pass of the ``QuestionDrivenController``

        Args:
            step (int): index of the current reasoning step
            contextual_words (Tensor): [batch_size x maxQuestionLength x dim]
                Word encodings
            question_encoding (Tensor): [batch_size x (2*dim)]
                Global representation of sentence
            ctrl_state (Tensor): [batch_size x dim]
                Previous control state

        Returns:
            Tensor: [batch_size x dim]
                New control state
            Tensor: [batch_size x num_temporal]
                Temporal class weights of current reasoning step
        """

        # pass question encoding through current 'position aware' linear layer
        pos_aware_question_encoding = self.pos_aware_layers[step](
            question_encoding)

        # concat control state and position aware question encoding
        cqi = torch.cat([ctrl_state, pos_aware_question_encoding], dim=-1)

        # project from 2dim to 1dim
        cqi = self.ctrl_question(cqi)  # [batch_size x dim]

        # retrieve content new_control_state + attention control_attention
        new_ctrl_state, ctrl_attention = self.attention_module(
            cqi, contextual_words)

        # neural network  that returns temporal class weights
        temporal_weights = self.temporal_classifier(new_ctrl_state)

        # return control and the temporal class weights
        return new_ctrl_state, ctrl_attention, temporal_weights
