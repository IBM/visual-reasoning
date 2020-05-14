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
import json
from skimage import io

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from torchvision.transforms import transforms

from nemo.backends.pytorch.nm import DataLayerNM
from nemo.core.neural_types import ChannelType, NeuralType, LengthsType, LabelsType
from nemo.utils.decorators import add_port_docs
from nemo.collections.nlp.data import WordTokenizer

class CLEVRDataset(Dataset):

    def __init__(self, root, split_name: str, transform=None):
        super().__init__()

        self._root = root
        self._split_name = split_name
        self.transform = transform

        # Create vocabulary if it hasn't been done already
        if not os.path.isfile("data/CLEVR_v1.0/questions/vocab_questions_words.txt") \
            or not os.path.isfile("data/CLEVR_v1.0/questions/vocab_answers_sentences.txt"):

            print("Building CLEVR vocabulary...")

            with open("data/CLEVR_v1.0/questions/CLEVR_train_questions.json", "r") as f:
                self._questions_data = json.load(f)["questions"]

            def make_vocab_words(sentences, output_path):
                vocab = set()
                for item in sentences:
                    tokens = item.strip().split()
                    vocab.update(tokens)
                with open(output_path, "w") as f:
                    for item in vocab:
                        f.write(f"{item}\n")

            def make_vocab_sentences(sentences, output_path):
                vocab = set()
                for item in sentences:
                    vocab.update([item])
                with open(output_path, "w") as f:
                    for item in vocab:
                        f.write(f"{item}\n")

            make_vocab_words(
                map(lambda item: item["question"], self._questions_data),
                "data/CLEVR_v1.0/questions/vocab_questions_words.txt"
                )

            make_vocab_sentences(
                map(lambda item: item["answer"], self._questions_data),
                "data/CLEVR_v1.0/questions/vocab_answers_sentences.txt"
                )

        assert split_name in ["train", "val", "test"]
        with open(f"data/CLEVR_v1.0/questions/CLEVR_{split_name}_questions.json", "r") as f:
            self._questions_data = json.load(f)["questions"]

    def __len__(self):
        return len(self._questions_data)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        
        data = self._questions_data[index]

        img_name = os.path.join(self._root, f"images/{self._split_name}", data["image_filename"])
        image = io.imread(img_name)

        # Remove alpha channel
        image = image[:, :, :3]

        if self.transform:
            image = self.transform(image)

        return image, data["question"], data["answer"]


class CLEVRDataLayer(DataLayerNM):

    @property
    @add_port_docs()
    def output_ports(self):
        """Returns definitions of module output ports.
        """
        return {
            "image": NeuralType(('B', 'C', 'H', 'W'), ChannelType()),
            "question": NeuralType(('B', 'T'), ChannelType()),
            # "question_len": NeuralType(tuple('B'), LengthsType()),
            "answer": NeuralType(('B'), LabelsType()),
        }

    def __init__(self, batch_size, root, split_name, shuffle=True):
        super().__init__()

        self._input_size = (3, 480, 320)

        self._batch_size = batch_size
        assert split_name in ["train", "val", "test"]
        self._split_name = split_name
        self._shuffle = shuffle
        self._root = root
        self._transforms = transforms.Compose([transforms.ToTensor()])

        self._dataset = CLEVRDataset(
            root=self._root, 
            split_name=self._split_name, 
            transform=self._transforms
            )

        answer_labels = open("data/CLEVR_v1.0/questions/vocab_answers_sentences.txt", "r").readlines()
        self.answer_labels = {answer_labels[i].strip(): i for i in range(len(answer_labels))}

        self.label_to_ix = {label: i for i, label in enumerate(self.answer_labels)}
        self.ix_to_label = {v: k for k, v in self.label_to_ix.items()}

        class DataserWrapper(Dataset):
            def __init__(self, dataset, answer_label_to_ix):
                super().__init__()
                self._dataset = dataset
                self._answer_label_to_ix = answer_label_to_ix

            def __getitem__(self, index):
                image, question, answer = self._dataset[index]
                answer = self._answer_label_to_ix[answer]
                return image, question, answer

            def __len__(self):
                return len(self._dataset)

        self._dataset = DataserWrapper(self._dataset, self.label_to_ix)


    def __len__(self):
        return len(self._dataset)

    @property
    def dataset(self):
        return self._dataset

    @property
    def data_iterator(self):
        return None
