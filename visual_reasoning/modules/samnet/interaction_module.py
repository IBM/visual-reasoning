# Copyright (C) IBM Corporation 2020
#
# SPDX-License-Identifier: Apache-2.0

__author__ = "T.S. Jayram"

import torch
from torch import nn
from visual_reasoning.modules import Linear


class InteractionModule(nn.Module):
    """
    ``Interaction Module``: Modulates feature objects with a base object
    """

    def __init__(self, dim, do_project=True):
        """Constructor for the ``InteractionModule``

        Args:
            dim (int): common dimension of base and feature objects
            do_project (bool, optional): flag indicating whether we need to
                project the feature objects. If False, then the projected
                feature objects is specified as an additional argument.
                Defaults to True
        """

        # call base constructor
        super(InteractionModule, self).__init__()

        # linear layer to project of the query
        self.base_layer = Linear(dim, dim)

        # linear layer for the projection of the keys
        self.feature_layer = Linear(dim, dim) if do_project else None

        self.do_project = do_project

        # linear layer to modulate feature objects by base object
        self.modulator = Linear(2 * dim, dim, bias=True)

    def forward(self, base, features, features_proj=None):
        """Forward pass of the ``InteractionModule``

        Args:
            base (Tensor): [batch_size x dim]
                Base object
            features (Tensor): [batch_size x num_objects x dim]
                Feature objects
            features_proj (Tensor, optional): [batch_size x num_objects x dim]
                Projected feature objects. Defaults to None.

        Returns:
            Tensor: [batch_size x num_objects x dim]
                Feature objects modulated by the base object
        """

        if self.do_project:
            assert features_proj is None
        else:
            assert features.size() == features_proj.size(), (
                'Shape mismatch between feature objects and their projection')

        # pass query object through linear layer
        base_proj = self.base_layer(base)
        # [batch_size x dim]

        # pass feature_objects through linear layer
        if self.do_project:
            features_proj = self.feature_layer(features)
        # [batch_size x num_objects x dim]

        # modify the projected feature objects using the projected base object
        feature_obj_modified = torch.cat([
            base_proj[:, None, :] * features_proj,
            features], dim=-1)

        features_modified = self.modulator(feature_obj_modified)
        # [batch_size x num_objects x dim]

        return features_modified
