# Copyright (C) IBM Corporation 2020
#
# SPDX-License-Identifier: Apache-2.0


import os
import argparse
from itertools import chain
from statistics import mean
from datetime import datetime
import socket

from nemo.utils.argparse import NemoArgParser
from nemo.core import (
    NeuralModuleFactory,
    DeviceType,
    NeuralGraph,
    OperationMode,
    SimpleLossLoggerCallback,
    SimpleLogger,
    NeMoCallback,
)
from nemo.core.deprecated_callbacks import EvaluatorCallback
from nemo.core import CheckpointCallback

from nemo.utils import logging
from nemo.core.neural_types import NmTensor
from nemo.utils.app_state import AppState

from nemo.collections.visual_reasoning.modules import COGDataLayer
from nemo.collections.cv.modules.non_trainables import NonLinearity
from visual_reasoning.modules.samnet import QuestionEncoder, SAMNet
from visual_reasoning.modules import ImageEncoder, NLLLoss

from nemo.collections.visual_reasoning.utils.cog_stats import (
    calculate_stats,
    collate_stats,
    calculate_accuracies,
)


def main():
    parser = argparse.ArgumentParser(
        parents=[NemoArgParser()], conflict_handler="resolve"
    )
    args = parser.parse_args()

    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    log_dir = os.path.join('runs', current_time + '_' + socket.gethostname())
    os.makedirs(log_dir, exist_ok=False)

    nf = NeuralModuleFactory(placement=DeviceType.GPU)

    if args.create_tb_writer:
        from torch.utils.tensorboard import SummaryWriter

        tb_writer = SummaryWriter(log_dir=log_dir)
    else:
        tb_writer = None

    ### INSTANTIATE THE NEURAL MODULES ###

    cog_datalayer_train = COGDataLayer(
        subset="train",
        cog_tasks="class",
        cog_type="canonical",
        batch_size=48,
        shuffle=True,
        pin_memory=True,
    )

    cog_datalayer_val = COGDataLayer(
        subset="val",
        cog_tasks="class",
        cog_type="canonical",
        batch_size=256,
        pin_memory=True,
    )

    dim = 128

    question_encoder = QuestionEncoder(
        vocabulary_size=cog_datalayer_train.tokenizer.vocab_size,
        embedded_dim=64,
        dim=dim,
        padding_idx=cog_datalayer_train.tokenizer.pad_id,
    ).cuda()

    image_encoder = ImageEncoder(dim=dim).cuda()

    samnet = SAMNet(
        dim=dim,
        max_step=8,
        num_temporal=4,
        dropout_factor=0.15,
        slot=8,
        num_outputs=len(cog_datalayer_train.output_vocab),
    ).cuda()

    logsoftmax = NonLinearity(type="logsoftmax", sizes=[-1, None, None])

    nll_loss = NLLLoss(input_dims=[-1, None, None], ignore_index=-1)

    ### TRAINING GRAPH ###

    with NeuralGraph(operation_mode=OperationMode.training) as training_graph:
        # We're ignoring all the pointing tasks
        (
            images,
            tasks,
            questions,
            questions_lens,
            _,
            targets_answer,
            _,
            mask_words,
            _,
            _,
        ) = cog_datalayer_train()

        encoded_image = image_encoder(images=images)

        contextual_words, question_encoding = question_encoder(
            questions=questions, questions_len=questions_lens
        )

        logits_answer = samnet(
            images_encoding=encoded_image,
            contextual_words=contextual_words,
            question_encoding=question_encoding,
        )

        logprobs = logsoftmax(inputs=logits_answer)

        loss = nll_loss(predictions=logprobs, targets=targets_answer)

        training_graph.outputs["loss"] = loss

    # Display the graph summmary.
    logging.info(training_graph.summary())

    ### VALIDATION GRAPH ###

    with NeuralGraph(operation_mode=OperationMode.evaluation) as eval_graph:
        # We're ignoring all the pointing tasks
        (
            images_eval,
            tasks_eval,
            questions_eval,
            questions_lens_eval,
            _,
            targets_answer_eval,
            _,
            mask_words_eval,
            _,
            _,
        ) = cog_datalayer_val()
        tasks_eval.rename("tasks_eval")
        mask_words_eval.rename("mask_words_eval")
        targets_answer_eval.rename("targets_answer_eval")

        encoded_image_eval = image_encoder(images=images_eval)

        contextual_words_eval, question_encoding_eval = question_encoder(
            questions=questions_eval, questions_len=questions_lens_eval
        )

        logits_answer_eval = samnet(
            images_encoding=encoded_image_eval,
            contextual_words=contextual_words_eval,
            question_encoding=question_encoding_eval,
        )
        logits_answer_eval.rename("logits_answer_eval")

        logprobs_eval = logsoftmax(inputs=logits_answer_eval)
        logprobs_eval.rename("logprobs_eval")

        loss_eval = nll_loss(predictions=logprobs_eval, targets=targets_answer_eval)
        loss_eval.rename("loss_eval")

    ### CALLBACKS ###

    def train_tb_func(tb_writer, tensors, step: int):
        loss = tensors[0].item()
        tb_writer.add_scalar("Loss/train", loss, step)

    # SimpleLossLoggerCallback will print loss values to console.
    callback = SimpleLossLoggerCallback(
        tensors=[loss],
        print_func=lambda x: logging.info(f"Training Loss: {str(x[0].item())}"),
        tb_writer=tb_writer,
        log_to_tb_func=train_tb_func,
    )

    # Computing the stats for each batch, storing in global var list
    def eval_loss_per_batch_callback(tensors, global_vars):
        if "eval_stats" not in global_vars.keys():
            global_vars["eval_stats"] = []
            global_vars["loss"] = []

        tasks = tensors[AppState().tensor_names["tasks_eval"]][0]
        mask_words = tensors[AppState().tensor_names["mask_words_eval"]][0]
        logits = tensors[AppState().tensor_names["logits_answer_eval"]][0]
        targets_answer = tensors[AppState().tensor_names["targets_answer_eval"]][0]
        loss = tensors[AppState().tensor_names["loss_eval"]][0]

        global_vars["eval_stats"] += [
            calculate_stats(
                tasks=tasks,
                mask_words=mask_words,
                prediction_answers=logits,
                target_answers=targets_answer,
            )
        ]

        global_vars["loss"] += [loss.cpu().item()]

    # Collate the stats from all the validation batches, calculate accuracies, print to log
    def eval_loss_epoch_finished_callback(global_vars):

        stats_collated = collate_stats(global_vars["eval_stats"])
        accuracies = calculate_accuracies(stats_collated)
        logging.info(f"Validation accuracies: {accuracies}")
        global_vars["eval_stats"] = []

        loss = mean(global_vars["loss"])
        logging.info(f"Validation loss: {loss}")
        global_vars["loss"] = []

        return accuracies, loss

    def eval_tb_func(tb_writer, values: (dict, float), step: int):
        accuracies, loss = values
        tb_writer.add_scalar("Loss/val", loss, step)

        for k, v in accuracies.items():
            tb_writer.add_scalar(f"Accuracy/val/{k}", v, step)

    # Validation callback, calling the 2 functions above accordingly
    ecallback = EvaluatorCallback(
        eval_tensors=[
            tasks_eval,
            mask_words_eval,
            logprobs_eval,
            logits_answer_eval,
            targets_answer_eval,
            loss_eval,
        ],
        user_iter_callback=eval_loss_per_batch_callback,
        user_epochs_done_callback=eval_loss_epoch_finished_callback,
        eval_step=10000,
        tb_writer=tb_writer,
        tb_writer_func=eval_tb_func,
    )

    ckpt_callback = CheckpointCallback(
        folder=log_dir,
        step_freq=10000,
        checkpoints_to_keep=1000
    )

    ### TRAIN ###

    # Invoke the "train" action.
    nf.train(
        training_graph=training_graph,
        callbacks=[callback, ecallback, ckpt_callback],
        optimization_params={"num_epochs": 5, "lr": 0.0002},
        optimizer="adam",
    )


if __name__ == "__main__":
    main()
