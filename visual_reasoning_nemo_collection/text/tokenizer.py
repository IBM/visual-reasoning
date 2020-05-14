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
from torch import nn
from nemo.backends.pytorch import NonTrainableNM
from nemo.core.neural_types import NeuralType, ChannelType, LengthsType
from nemo.utils.decorators import add_port_docs
from nemo.collections.nlp.data import WordTokenizer

class Tokenizer(NonTrainableNM):
    @property
    @add_port_docs()
    def input_ports(self):
        """Returns definitions of module input ports.
        """
        return {
            "x": NeuralType(('B', 'T'), ChannelType()),
        }

    @property
    @add_port_docs()
    def output_ports(self):
        """Returns definitions of module output ports.
        """
        return {
            "output": NeuralType(('B', 'T'), ChannelType()),
            "output_len": NeuralType(tuple('B'), LengthsType()),
        }

    def __init__(self, vocab_path: str):
        super().__init__()

        assert os.path.isfile(vocab_path)
        self._tokenizer = WordTokenizer(vocab_path)

    def forward(self, x):
        tokenized = [self._tokenizer.text_to_ids(elem) for elem in x]
        output_lens = [len(elem) for elem in tokenized]
        # Sort in decreasing length order to make pack_padded_sequence happy
        output_lens, tokenized = (list(e) for e in zip(*sorted(zip(output_lens, tokenized), reverse=True)))
        tokenized = [torch.LongTensor(elem) for elem in tokenized]

        tokenized_padded = nn.utils.rnn.pad_sequence(tokenized, batch_first=True, padding_value=self._tokenizer.vocab['<PAD>'])
        tokenized_padded = torch.tensor(tokenized_padded, dtype=torch.long).to(self._device)

        return tokenized_padded, output_lens
