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

import os
import torch
from nemo.backends.pytorch import NonTrainableNM, TrainableNM
from nemo.core.neural_types import NeuralType, ChannelType, VoidType, LogprobsType
from nemo.utils.decorators import add_port_docs
from nemo.core import DeviceType

class LogSoftmax(TrainableNM):
    @property
    @add_port_docs()
    def input_ports(self):
        """Returns definitions of module input ports.
        """
        return {
            "x": NeuralType(None, VoidType()),
        }

    @property
    @add_port_docs()
    def output_ports(self):
        """Returns definitions of module output ports.
        """
        return {
            "output": NeuralType(('B'), LogprobsType()),
        }

    def __init__(self, dim=1):
        super().__init__()

        self.logsoftmax = torch.nn.LogSoftmax(dim=dim)

        self._device = torch.device("cuda" if self.placement == DeviceType.GPU else "cpu")
        self.to(self._device)

    def forward(self, x):
        return self.logsoftmax(x)
