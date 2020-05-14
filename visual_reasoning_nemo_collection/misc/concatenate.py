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
from nemo.core.neural_modules import NeuralModule
from nemo.backends.pytorch import NonTrainableNM
from nemo.core.neural_types import NeuralType
from nemo.utils.decorators import add_port_docs
from nemo.collections.nlp.data import WordTokenizer
from collections import OrderedDict

class Concatenate(NonTrainableNM):
    @property
    @add_port_docs()
    def input_ports(self):
        """Returns definitions of module input ports.
        """
        return self._input_ports

    @property
    @add_port_docs()
    def output_ports(self):
        """Returns definitions of module output ports.
        """
        return self._output_ports

    def __init__(self, dim: int, num_inputs: int, neural_type: NeuralType):
        super().__init__()

        assert num_inputs > 1

        self._concat_dim = dim

        self._input_ports = OrderedDict()
        for i in range(num_inputs):
            self._input_ports[f"x_{i}"] = neural_type

        self._output_ports = {
            "output": neural_type
        }

    def forward(self, **kwargs):
        tensor_list = []
        for k in self._input_ports.keys():
            tensor_list += [kwargs[k]]

        return torch.cat(tuple(tensor_list), dim=self._concat_dim)
