# Copyright (C) IBM Corporation 2020
#
# SPDX-License-Identifier: Apache-2.0

__author__ = "T.S. Jayram"

import torch
from torch import nn
from .utils import linear


class AttentionModule(nn.Module):
    """
    ``Attention Module``: Using weighted cosine similarity
    """

    def __init__(self, dim):
        """Constructor for the Attention Module

        Args:
            dim (int): common dimension of query vector and keys
        """

        # call base constructor
        super(AttentionModule, self).__init__()

        # define a linear layer for the attention weights.
        self.attn = torch.nn.Sequential(
            linear(dim, 1, bias=False),
            torch.nn.Softmax(dim=1))
        self.dim = dim

    def forward(self, query, keys, values=None):
        """Forward pass of the ``Attention_Module``

        Args:
            query (Tensor): [batch_size x dim]
            keys (Tensor): [batch_size x N x dim]
            values (Tensor, optional): [batch_size x N x dim_other]
                Defaults to the same value as keys.

        Returns:
            Tensor: [batch_size x dim_other]
                Content vector
            Tensor: [batch_size x N]
                Attention vector
        """

        if values is None:
            values = keys

        assert query.size(-1) == self.dim, 'Dimension mismatch in query'
        assert keys.size(-1) == self.dim, 'Dimension mismatch in keys'
        assert values.size(-2) == keys.size(-2), (
            'Number of entities mismatch between keys and values')

        # compute element-wise weighted product between query and keys
        # and normalize them

        ca = self.attn(query[:, None, :] * keys)  # [batch_size x N x 1]

        # compute content to be retrieved
        c = (ca * values).sum(1)     # [batch_size x dim_other]

        ca = ca.squeeze(-1)     # [batch_size x N]

        return c, ca
