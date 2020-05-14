# Copyright (C) IBM Corporation 2020
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from torch import nn
from nemo.backends.pytorch.nm import TrainableNM
from nemo.core.neural_types import ChannelType, NeuralType
from nemo.utils.decorators import add_port_docs
from nemo.core import DeviceType

class MultiLayerPerceptron(TrainableNM):
    """
    If only input and output dims are given, will be single layer (no hidden layers).
    If hidden_dims (list of ints) is also given, hidden layers are added along with the given `hidden_nonlinearity`.
    `last_nonlinearity` is optional, to add non linearity after the last layer.
    """

    @property
    @add_port_docs()
    def input_ports(self):
        """Returns definitions of module input ports.
        """
        return {
            "x": NeuralType(('B', 'D'), ChannelType()),
        }

    @property
    @add_port_docs()
    def output_ports(self):
        """Returns definitions of module output ports.
        """
        return {
            "output": NeuralType(('B', 'D'), ChannelType()),
        }

    def __init__(self, input_dim, output_dim, hidden_dims: list=[], hidden_nonlinearities=nn.ReLU, last_nonlinearity=None):
        super().__init__()

        dims = [input_dim] + hidden_dims + [output_dim]

        module_list = []

        for dim_in, dim_out in zip(dims, dims[1:]):
            module_list += [nn.Linear(dim_in, dim_out)]
            if hidden_nonlinearities:
                module_list += [hidden_nonlinearities()]

        # Replace the last nonlinearity
        if last_nonlinearity:
            module_list[-1] = last_nonlinearity()
        else: # if None
            del module_list[-1]

        self.model = nn.Sequential(*module_list)

        self._device = torch.device("cuda" if self.placement == DeviceType.GPU else "cpu")
        self.to(self._device)

    def forward(self, x):
        return self.model(x)
