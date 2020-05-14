# Copyright (C) IBM Corporation 2020
#
# SPDX-License-Identifier: Apache-2.0

__author__ = "T.S. Jayram"

import torch
from torch import nn

from .utils import linear


class OutputUnit(nn.Module):
    """
    ``OutputUnit``
    """

    def __init__(self, dim, num_outputs):
        """
        Constructor for ``OutputUnit``

       Args:
            dim (int): dimension of tensor representing
                the features to be classified
            num_outputs (int): number of output classes
        """

        # call base constructor
        super(OutputUnit, self).__init__()

        # define the 2-layers MLP
        self.classifier = torch.nn.Sequential(
            linear(dim, dim, bias=True, initializer=nn.init.kaiming_uniform_),
            torch.nn.ELU(),
            linear(dim, num_outputs, bias=True))

    def forward(self, features):
        """Forward pass of ``OutputUnit``

        Args:
            features (Tensor): [batch_size x dim] Features to be classified

        Returns:
            Tensor: logits for the classes
        """

        logits = self.classifier(features)

        return logits
