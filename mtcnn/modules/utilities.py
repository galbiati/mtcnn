#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# By: Gianni Galbiati

# Standard libraries

# External libraries
import numpy as np
import torch
import torch.nn as nn

# Internal libraries


def load_from_numpy(module, weight_path):
    """Load weights stored in npy format to a torch neural network module."""

    weights = np.load(weight_path, allow_pickle=True)[()]

    for n, p in module.named_parameters():
        p.data = torch.as_tensor(weights[n], dtype=torch.float32)

    return None


class Flatten(nn.Module):
    """Convenience class that flattens feature maps to vectors."""
    def forward(self, x):
        # Required for pretrained weights for some reason?
        x = x.transpose(3, 2).contiguous()

        return x.view(x.size(0), -1)
