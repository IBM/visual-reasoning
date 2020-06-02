# Copyright (C) IBM Corporation 2020
#
# SPDX-License-Identifier: Apache-2.0

__author__ = "T.S. Jayram"

import torch


def memory_update(memory, visual_object, read_head, write_head,
                  do_replace, do_add_new):
    """``Memory Update Unit``: Update the memory based on new information

    Args:
        memory (Tensor): [batch_size x mem_size x dim]
        visual_object (Tensor): [batch_size x dim]
        read_head (Tensor): [batch_size x mem_size]
        write_head (Tensor): [batch_size x mem_size]
        do_replace (Tensor): [batch_size]
        do_add_new (Tensor): [batch_size]

    Returns:
        Tensor: [batch_size x mem_size x dim]
            Updated memory
        Tensor: [batch_size x mem_size]
            Updated write head
    """

    # pad extra dimension
    do_replace = do_replace[..., None]
    do_add_new = do_add_new[..., None]

    # get attention on the correct slot in memory based on the 2 predicates
    # attention defaults to 0 if neither condition holds
    wt = do_replace * read_head + do_add_new * write_head

    # Update visual_working_memory
    new_memory = (memory * (1 - wt)[..., None]
                  + wt[..., None] * visual_object[..., None, :])

    # compute shifted sequential head to right
    shifted_write_head = torch.roll(write_head, shifts=1, dims=-1)

    # new sequential attention
    new_write_head = ((shifted_write_head * do_add_new)
                      + (write_head * (1 - do_add_new)))

    return new_memory, new_write_head
