# Copyright (C) IBM Corporation 2020
#
# SPDX-License-Identifier: Apache-2.0

__author__ = "T.S. Jayram"

import torch
from torch import nn

from nemo.backends.pytorch import TrainableNM
from nemo.utils.decorators import add_port_docs
from nemo.core import NeuralType, EmbeddedTextType, EncodedRepresentation, TokenIndex, LengthsType


class QuestionEncoder(TrainableNM):
    """
    Implementation of the ``QuestionEncoder`` of the VWM network.
    """

    def __init__(self, vocabulary_size, embedded_dim, dim, padding_idx):
        """Constructor for ``QuestionEncoder``

        Args:
            vocabulary_size (int): size of dictionnary
            embedded_dim (int): dimension of the word embeddings
            dim (int): common dimension of all objects
        """

        # call base constructor
        super(QuestionEncoder, self).__init__()

        # create bidirectional LSTM layer
        self.lstm = torch.nn.LSTM(
            input_size=embedded_dim, hidden_size=dim,
            num_layers=1, batch_first=True, bidirectional=True)

        # linear layer for projecting the word encodings from 2*dim to dim
        self.lstm_proj = torch.nn.Linear(2 * dim, dim)

        # Defines nn.Embedding for embedding of questions into float tensors.
        self.embedding = nn.Embedding(
            vocabulary_size, embedded_dim, padding_idx=padding_idx)

    @property
    @add_port_docs
    def input_ports(self):
        """Returns definition of the module's input ports"""

        return {
            "questions": NeuralType(('B', 'T'), TokenIndex()), # Sentence token ids
            "questions_len": NeuralType(('B',), LengthsType())
        }

    @property
    @add_port_docs
    def output_ports(self):
        """Returns definition of the module's output ports"""

        return {
            "contextual_word_embedding": NeuralType(('B', 'T', 'D'), EmbeddedTextType()),
            "question_encoding": NeuralType(('B', 'D'), EncodedRepresentation()),            
        }

    def forward(self, questions, questions_len):
        """Forward pass of ``QuestionEncoder``

        Args:
            questions (Tensor): [batch_size x maxQuestionLength x embedded_dim]
                Question words as feature vectors

            questions_len (int): Unpadded questions length

        Returns:
            Tensor: [batch_size x (2*dim)]
                Question encodings
            Tensor: [batch_size x maxQuestionLength x dim]
                Contextual_word_embedding
        """

        # get batch size
        batch_size = questions.shape[0]

        # Embeddings.
        embedded_questions = self.embedding(questions)

        # LSTM layer: words & questions encodings
        lstm_out, (h, _) = self.lstm(embedded_questions)

        # get final words encodings using linear layer
        contextual_word_embedding = self.lstm_proj(lstm_out)

        # reshape last hidden states for questions encodings
        # -> [batch_size x (2*dim)]
        question_encoding = (h.permute(1, 0, 2)
                             .contiguous()
                             .view(batch_size, -1))

        return contextual_word_embedding, question_encoding
