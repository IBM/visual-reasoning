# Copyright (C) IBM Corporation 2020
#
# SPDX-License-Identifier: Apache-2.0

import torch

from nemo.backends.pytorch.nm import LossNM
from nemo.core.neural_types import LabelsType, LogprobsType, LossType, NeuralType
from nemo.utils.decorators import add_port_docs

__all__ = ['NLLLoss']


class NLLLoss(LossNM):
    """ Class representing a simple NLL loss. """

    def __init__(self, input_dims=[-1], ignore_index=-100):
        # Call the base class constructor.
        super().__init__()
        self._input_dims = input_dims
        self._nllloss = torch.nn.NLLLoss(ignore_index=ignore_index)

    @property
    @add_port_docs()
    def input_ports(self):
        """ Returns definitions of module input ports. """
        axes_preds = tuple(['B'] + ['Any' for _ in self._input_dims[1:]])
        axes_tgts = tuple(['B'] + ['Any' for _ in self._input_dims[2:]])
        return {
            "predictions": NeuralType(axes=axes_preds, elements_type=LogprobsType()),
            "targets": NeuralType(axes=axes_tgts, elements_type=LabelsType()),
        }

    @property
    @add_port_docs()
    def output_ports(self):
        """ Returns definitions of module output ports. """
        return {"loss": NeuralType(elements_type=LossType())}

    def _loss_function(self, predictions, targets):
        # Flatten all but last dim. Ex: (B, T, D) -> (B*T, D)
        predictions = predictions.reshape((-1, predictions.size()[-1]))
        # Flatten targets. Ex: (B, T) -> (B*T)
        targets = targets.flatten()

        return self._nllloss(input=predictions, target=targets)
