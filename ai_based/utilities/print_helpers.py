from typing import Dict

import torch
import numpy as np


def pretty_print_dict(dict_: Dict, indent_tabs: int = 0, line_prefix: str = "+ "):
    prefix_str = "\t" * indent_tabs
    if line_prefix != "":
        prefix_str += line_prefix
    for key_, val_ in dict_.items():
        if (isinstance(val_, torch.Tensor) and val_.dim() == 1) or (isinstance(val_, np.ndarray) and val_.ndim == 1):
            vector_str = "   ".join([f"{v_:.4f}" for v_ in val_])
            print(prefix_str + f"{key_}: {vector_str}")
        elif (isinstance(val_, torch.Tensor) and val_.dim() != 0) or (isinstance(val_, np.ndarray) and val_.ndim != 0):
            print(prefix_str + f"{key_}: {val_}")
        elif isinstance(val_, dict):
            print(prefix_str + f"{key_}:")
            pretty_print_dict(val_, indent_tabs=indent_tabs+1)
        else:
            val_str = f"{val_:,}" if isinstance(val_, int) else f"{val_:.4f}"
            print(prefix_str + f"{key_}: {val_str}")
