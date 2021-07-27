import math
import torch
import torch.nn as nn

class sin_act(nn.Module):
    """
    sinusoidal activation function.
    """
    def __init__(self):
        super(sin_act, self).__init__()

    def forward(self, x):
        return torch.sin(x)

class linear_act(nn.Module):
    """
    linear activation function.
    """
    def __init__(self):
        super(linear_act, self).__init__()

    def forward(self, x):
        return x

class mish(nn.Module):
    """
    mish activation function.
    """
    def __init__(self):
        super(mish, self).__init__()
    def forward(self, x):
        return x * torch.tanh(nn.functional.softplus(x))

class gelu_fast(nn.Module):
    """
    gelu_fast activation function.  
    """
    def __init__(self):
        super(gelu_fast, self).__init__()
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(x * 0.7978845608 * (1.0 + 0.044715 * x * x)))

class gelu_new(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT). Also see
    the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """
    def __init__(self):
        super(gelu_new, self).__init__()
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

def get(identifier):
    """
    get activation function.
    """
    return{
            "elu": nn.ELU(),
            "relu": nn.ReLU(),
            "sigmoid": nn.Sigmoid(),
            "leaky": nn.LeakyReLU(),
            "tanh": nn.Tanh(),
            "sin": sin_act(),
            "linear": linear_act(),
            "silu": nn.SiLU(),
            "mish": mish(),
            "gelu-fast": gelu_fast(),
            "gelu-new": gelu_new(),
    }[identifier]