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
import nemo
from nemo.core.neural_types import NeuralType, ChannelType
from nemo.core import NeuralGraph, OperationMode, DeviceType
from nemo.utils import logging
from visual_reasoning_nemo_collection.data import CLEVRDataLayer
from visual_reasoning_nemo_collection.image import Resnet50
from visual_reasoning_nemo_collection.text import Tokenizer
from visual_reasoning_nemo_collection.loss import NLLLoss
from visual_reasoning_nemo_collection.misc import EncoderRNN, Concatenate, MultiLayerPerceptron, LogSoftmax

# Define model parameters
batch_size = 32
data_root = "data/CLEVR_v1.0"

neural_factory = nemo.core.NeuralModuleFactory(
    log_dir='logs',
    create_tb_writer=False,
    placement=DeviceType.GPU
)

clevr_data_train = CLEVRDataLayer(
    batch_size=batch_size, 
    root=data_root, 
    split_name="train"
)
clevr_data_val = CLEVRDataLayer(
    batch_size=batch_size, 
    root=data_root, 
    split_name="val"
)

# Resnet50: load weights from torchvision, freeze all but last FC layer
resnet = Resnet50(output_size=512)
resnet.freeze()
resnet.unfreeze(set(["fc.weight", "fc.bias"]))

# Tokenize (text->ids) + pad
tokenizer = Tokenizer(vocab_path="data/CLEVR_v1.0/questions/vocab_questions_words.txt")

# Encode question through LSTM into 512-long vector (last output)
question_encoder = EncoderRNN(
    input_dim=tokenizer._tokenizer.vocab_size,
    emb_dim=512,
    hid_dim=512,
    dropout=0,
    n_layers=1,
    output_last_only=True # we only want the last timestep output
)

# Concatenate into 1024 long vector
concat = Concatenate(
    dim=1,
    num_inputs=2,
    neural_type=NeuralType(('B', 'D'), ChannelType())
)

# 3-layer NLP, with dims 1024 -> 256 -> #answer_categories
mlp = MultiLayerPerceptron(
    input_dim=1024,
    output_dim=len(clevr_data_train.answer_labels),
    hidden_dims=[256],
    hidden_nonlinearities=torch.nn.Tanh
)

logsoftmax = LogSoftmax(dim=1)

nllloss = NLLLoss()

# 2. Create a training graph.
with NeuralGraph(operation_mode=OperationMode.training) as training_graph:
    # CLEVR training data
    image, question, answer = clevr_data_train()

    # Image pipeline
    encoded_image = resnet(x=image)

    # Question nlp pipeline
    tokenized_question, question_len = tokenizer(x=question)
    encoded_question, _ = question_encoder(inputs=tokenized_question, input_lens=question_len)

    # Concatenate image and question encoded vectors, pass through 3 layer MLP
    concat_image_question = concat(x_0=encoded_image, x_1=encoded_question)
    mlp_output = mlp(x=concat_image_question)

    # Compute logprobs, loss
    answer_inferred = logsoftmax(x=mlp_output)
    loss = nllloss(predictions=answer_inferred, targets=answer)

    # Set output - that output will be used for training.
    training_graph.outputs["loss"] = loss


# 3. Create a validation graph, starting from the second data layer.
with NeuralGraph(operation_mode=OperationMode.evaluation) as evaluation_graph:
    # CLEVR validation data
    image, question, answer = clevr_data_val()
    
    # Image pipeline
    encoded_image = resnet(x=image)
    
    # Question nlp pipeline
    tokenized_question, question_len = tokenizer(x=question)
    encoded_question, _ = question_encoder(inputs=tokenized_question, input_lens=question_len)
    
    # Concatenate image and question encoded vectors, pass through 3 layer MLP
    concat_image_question = concat(x_0=encoded_image, x_1=encoded_question)
    mlp_output = mlp(x=concat_image_question)
    
    # Compute logprobs, loss
    answer_inferred = logsoftmax(x=mlp_output)
    loss_e = nllloss(predictions=answer_inferred, targets=answer)


# 4. Create the callbacks.
def eval_loss_per_batch_callback(tensors, global_vars):
    if "eval_loss" not in global_vars.keys():
        global_vars["eval_loss"] = []
    for key, value in tensors.items():
        if key.startswith("loss"):
            global_vars["eval_loss"].append(torch.mean(torch.stack(value)))

def eval_loss_epoch_finished_callback(global_vars):
    eloss = torch.max(torch.tensor(global_vars["eval_loss"]))
    logging.info("Evaluation Loss: {0}".format(eloss))  
    return dict({"Evaluation Loss": eloss})

ecallback = nemo.core.EvaluatorCallback(
    eval_tensors=[loss_e],
    user_iter_callback=eval_loss_per_batch_callback,
    user_epochs_done_callback=eval_loss_epoch_finished_callback,
    eval_step=100,
)

# SimpleLossLoggerCallback will print loss values to console.
callback = nemo.core.SimpleLossLoggerCallback(
    tensors=[loss], print_func=lambda x: logging.info(f'Training Loss: {str(x[0].item())}')
)

# Invoke the "train" action.
neural_factory.train(training_graph=training_graph, callbacks=[callback, ecallback], optimization_params={"num_epochs": 10, "lr": 0.001}, optimizer="adam"
)
