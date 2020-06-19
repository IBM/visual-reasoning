# Copyright (C) IBM Corporation 2020
#
# SPDX-License-Identifier: Apache-2.0

__author__ = "T.S. Jayram"

import torch

class Linear(torch.nn.Linear):
    """Linear layer with user-defined weight initialization"""
    def __init__(
        self, 
        input_dim, 
        output_dim, 
        bias=True, 
        initializer=torch.nn.init.xavier_uniform_):
        """Args:
            input_dim (int): input dimension
            output_dim (int): output dimension
            bias (bool, optional): If set to True, train the bias parameter.
                Defaults to True
            initializer (torch.nn.init, optional): User-defined initializer.
                Defaults to Xavier uniform
        """
        super(Linear, self).__init__(input_dim, output_dim, bias=bias)

        # initialize weights
        initializer(self.weight)

        # initialize biases
        if bias:
            self.bias.data.zero_()
