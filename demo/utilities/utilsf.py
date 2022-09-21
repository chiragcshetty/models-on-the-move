import torch
import config
import ctypes, gc
import psutil, os
from datetime import datetime

## Estimate size of the model (in GB or MB or B)
## i.e sum of sizes of all parameters of the model
def estimate_model_size(model, unit='B', to_print = False): 
    persistent_memory = 0
    for name, param in model.named_parameters():
        persistent_memory += param.element_size() * param.nelement()
    if unit == 'GB':
        gb_mem = round(persistent_memory/1024**3,8)
        if to_print:
            print("Estimated Model Memory:",gb_mem, "GB")
        return gb_mem
    elif unit == 'B':
        gb_mem = persistent_memory
        if to_print:
            print("Estimated Model Memory:",gb_mem, "Bytes")
        return gb_mem
    else:
        mb_mem = round(persistent_memory/1024**2,8)
        if to_print:
            print("Estimated Model Memory:", mb_mem, "MB")
        return mb_mem


def estimate_tensor_size(inp, unit='B'):
    input_size = 0
    if isinstance(inp, torch.Tensor): 
        input_size += float(torch.prod(torch.tensor(inp.size())))
    elif isinstance(inp, list): 
        for sub_inp in inp:
            if isinstance(sub_inp, torch.Tensor): input_size += float(torch.prod(torch.tensor(sub_inp.size())))
    elif isinstance(inp, tuple): #for gnmt
        input_size += recursively_compute_tuple_mem(inp)
    elif inp is None:
        pass
    else:
        print(inp)
        raise ValueError("Something wrong here. Please handle this type:", type(inp))

    input_size = input_size*torch.rand((1,1)).element_size() # multiply by 4
    if unit == 'GB':
        gb_mem = round(input_size/1024**3,8)
        #print("Estimated Input/Output Memory:",gb_mem, "GB")
        return gb_mem
    if unit == 'B':
        gb_mem = input_size
        #print("Estimated Input/Output Memory:",gb_mem, "B")
        return gb_mem
    else:
        mb_mem = round(input_size/1024**2,8)
        #print("Estimated Input/Output Memory:", mb_mem, "MB")
        return mb_mem
