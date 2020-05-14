# Copyright (C) IBM Corporation 2020
#
# SPDX-License-Identifier: Apache-2.0

__author__ = "T.S. Jayram"

from torch import nn


def linear(input_dim, output_dim, bias=True,
           initializer=nn.init.xavier_uniform_):
    """Linear layer with user-defined weight initialization

    Args:
        input_dim (int): input dimension
        output_dim (int): output dimension
        bias (bool, optional): If set to True, train the bias parameter.
            Defaults to True
        initializer (torch.nn.init, optional): User-defined initializer.
            Defaults to Xavier uniform


    Returns:
        (torch.nn.Module): Initialized Linear layer
    """
    # define linear layer from torch.nn library
    linear_layer = nn.Linear(input_dim, output_dim, bias=bias)

    # initialize weights
    initializer(linear_layer.weight)

    # initialize biases
    if bias:
        linear_layer.bias.data.zero_()

    return linear_layer
