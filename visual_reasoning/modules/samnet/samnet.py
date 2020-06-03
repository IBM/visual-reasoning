# Copyright (C) IBM Corporation 2020
#
# SPDX-License-Identifier: Apache-2.0

"""
samnet.py:

    Implementation of the ``SAMNet`` model, using the different modules.
"""
__author__ = "T.S. Jayram"

import nltk

import numpy as np
import torch
import torch.nn as nn

from nemo.backends.pytorch import TrainableNM
from nemo.utils.decorators import add_port_docs
from nemo.core import DeviceType, NeuralType, ChannelType, VoidType, LogitsType

from visual_reasoning.modules import Linear, ImageEncoder
from . import (
    QuestionEncoder,
    QuestionDrivenController,
    SAMCell,
    memory_update,
    OutputUnit
)

# needed for nltk.word.tokenize - do it once!
nltk.download('punkt')


class SAMNet(TrainableNM):
    """
    Implementation of the entire ``SamNet`` network.
    """

    def __init__(
        self, 
        dim: int,
        embed_hidden: int,
        max_step: int,
        num_temporal: int,
        dropout_factor: float,
        slot: int,
        words_embed_length: int,
        num_outputs: int,
        vocabulary_size: int
    ):
        """Constructor for ``SAMNet``

        Args:
            params (dict): parameters read from configuration ``.yaml`` file
            problem_default_values_ (dict, optional): default values from
                the ``Problem`` class.
        """

        super(SAMNet, self).__init__()


        self.dim = dim
        self.embed_hidden = embed_hidden
        self.max_step = max_step
        self.num_temporal = num_temporal
        self.dropout_param = dropout_factor
        self.slot = slot
        self.words_embed_length = words_embed_length

        self.num_outputs = num_outputs
        # Maximum number of embeddable words.
        self.vocabulary_size = vocabulary_size

        self.name = 'SAMNet'

        # instantiate units
        self.question_encoder = QuestionEncoder(
            self.vocabulary_size, dim=self.dim, embedded_dim=self.embed_hidden)

        self.question_driven_controller = QuestionDrivenController(
            self.dim, self.max_step, self.num_temporal)

        # instantiate units
        self.image_encoder = ImageEncoder(dim=self.dim)

        # linear layer for the projection of image features
        self.feature_map_proj_layer = Linear(self.dim, self.dim, bias=True)

        # initialize SAMCell
        self.sam_cell = SAMCell(self.dim, self.num_temporal)

        # Create output units for classification
        self.output_unit_layer = OutputUnit(3*self.dim, self.num_outputs)

        # initialize hidden states for control state and summary object
        # this is also a trainable paremeter
        self.control_0 = torch.nn.Parameter(
            torch.zeros(1, self.dim))
        self.summary_obj_0 = torch.nn.Parameter(
            torch.zeros(1, self.dim))

        self.dropout_layer = torch.nn.Dropout(self.dropout_param)

    @property
    @add_port_docs
    def input_ports(self):
        """Returns definition of the module's input ports"""

        return {
            "images": NeuralType(('B', 'T', 'C', 'H', 'W'), ChannelType()),
            "question": NeuralType(('B', 'T'), VoidType()) # Sentence token ids
        }

    @property
    @add_port_docs
    def output_ports(self):
        """Returns definition of the module's output ports"""

        return {
            "answers": NeuralType(('B', 'T', 'D'), LogitsType)
        }


    def forward(self, images, questions):
        """Forward pass of ``SAMNet`` network.
            Calls first the ``ImageEncoder`` and ``QuestionEncoder``,
            then the recurrent SAMCells, and finally the ```OutputUnit``

        Args:
            data_dict (dict): Input data samples for this batch

        Returns:
            Tensor: Predictions of the model.
        """

        # print('============\nNew Run\n============')
        # Change the order of image dimensions, so we will loop over
        # dimension 0: sequence elements.
        images = images.permute(1, 0, 2, 3, 4)

        # Get batch size and length of image sequence.
        seq_len = images.size(0)
        batch_size = images.size(1)

        # Get questions size of all batch elements.
        questions_length = questions.size(1)

        # Convert questions length into a tensor
        questions_length = torch.from_numpy(np.array(questions_length))

        # Create placeholders for logits.
        logits_answer = torch.zeros(
            batch_size, seq_len, self.num_outputs)
        if self.placement is DeviceType.GPU or DeviceType.AllGpu:
            logits_answer = logits_answer.cuda()

        # Apply dropout to SAMCell control_state_init and summary_object states
        control_state_init = self.control_0.expand(batch_size, -1)
        control_state_init = self.dropout_layer(control_state_init)

        summary_object_init = self.summary_obj_0.expand(batch_size, -1)
        summary_object_init = self.dropout_layer(summary_object_init)

        # initialize empty memory
        memory = torch.zeros(batch_size, self.slot, self.dim)
        if self.placement is DeviceType.GPU:
            memory = memory.cuda()

        # initialize read head at first slot position
        write_head = torch.zeros(batch_size, self.slot)
        if self.placement is DeviceType.GPU:
            write_head = write_head.cuda()
        write_head[:, 0] = 1

        # question encoder
        contextual_words, question_encoding = self.question_encoder(
            questions, questions_length)

        control_state = control_state_init
        control_history = []
        for step in range(self.max_step):
            control_state, control_attention, temporal_weights = (
                self.question_driven_controller(
                    step, contextual_words, question_encoding, control_state))

            control_history.append(
                (control_state, control_attention, temporal_weights))

        # Loop over all elements along the SEQUENCE dimension.
        for f in range(images.size(0)):

            # RESET OF SUMMARY OBJECT
            summary_object = summary_object_init

            # image encoder
            feature_map = self.image_encoder(images[f])
            feature_map_proj = self.feature_map_proj_layer(feature_map)

            # state history
            sam_cell_hist = [None] * self.max_step

            # recurrent VWM cells
            for step in range(self.max_step):
                summary_object, sam_cell_info = self.sam_cell(
                    summary_object, control_history[step],
                    feature_map, feature_map_proj, memory)

                sam_cell_hist[step] = sam_cell_info

            # VWM update
            for step in range(self.max_step):
                # update VWM contents and write head
                visual_object = sam_cell_hist[step]['vo']
                read_head = sam_cell_hist[step]['rhd']
                do_replace = sam_cell_hist[step]['do_r']
                do_add_new = sam_cell_hist[step]['do_a']

                memory, write_head = memory_update(
                    memory, visual_object, read_head, write_head,
                    do_replace, do_add_new)

            # output unit
            logits_answer[:, f, :] = self.output_unit_layer(
                torch.cat([question_encoding, summary_object], dim=1))

        return logits_answer
