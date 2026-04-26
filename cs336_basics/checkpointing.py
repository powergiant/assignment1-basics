import torch
from torch.nn import Module
from torch.optim import Optimizer
import os, typing
import io

def save_checkpoint(model: Module, optimizer: Optimizer, iteration: int, 
                    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]):
    model_dict = model.state_dict()
    opt_dict = optimizer.state_dict()
    total_dict = {'model': model_dict, 'opt': opt_dict, 'it': iteration}

    torch.save(total_dict, out)




def load_checkpoint(src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes], 
                    model: Module, optimizer: Optimizer) -> int:
    total_dict = torch.load(src)
    model_dict: dict = total_dict['model']
    opt_dict: dict = total_dict['opt']
    iteration: int = total_dict['it']

    model.load_state_dict(model_dict)
    optimizer.load_state_dict(opt_dict)

    return iteration