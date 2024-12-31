import os
import tempfile

import torch
import torch.nn as nn


def compute_model_size(model: nn.Module) -> int:
    # see https://discuss.pytorch.org/t/finding-model-size/130275
    model_size = 0

    for p in model.parameters():
        model_size += p.numel() * p.element_size()

    for b in model.buffers():
        model_size += b.numel() * b.element_size()

    return model_size


def compute_state_dict_size(model: nn.Module) -> int:
    state_dict = model.state_dict()

    with tempfile.TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, "model.pth")
        torch.save(state_dict, path)
        file_size = os.path.getsize(path)

    return file_size
