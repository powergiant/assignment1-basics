import torch
from torch import Tensor
from torch.nn import Module, Parameter
from torch.nn.functional import silu
import math
from einops import einsum

class Linear(Module):
    def __init__(self, d_in: int, 
                 d_out: int, 
                 device: torch.device | None = None, 
                 dtype: torch.dtype | None = None):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.weight = Parameter(torch.empty(d_out, d_in, dtype=dtype, device=device))
        self._initial_weight()
        
    def _initial_weight(self):
        sigma = math.sqrt(2/(self.d_in + self.d_out))
        torch.nn.init.trunc_normal_(self.weight, 0.0, sigma, -3*sigma, 3*sigma)

    def forward(self, in_features: Tensor) -> Tensor:
        return in_features @ torch.transpose(self.weight, -1, -2)
        # return einsum(x, self.weight, '... in_feature, ... out_feature in_feature -> ... out_feature')

class Embedding(Module):
    def __init__(self, num_embd: int, 
                 d_embd: int, 
                 device: torch.device | None = None, 
                 dtype: torch.dtype | None = None):
        super().__init__()
        self.num_embd = num_embd
        self.d_embd = d_embd
        self.embd = Parameter(torch.empty(num_embd, d_embd, dtype=dtype, device=device))
        self._initial_embd()
        
    def _initial_embd(self):
        sigma = 1.0
        torch.nn.init.trunc_normal_(self.embd, 0.0, sigma, -3*sigma, 3*sigma)

    def forward(self, ids: Tensor) -> Tensor:
        return self.embd[ids]
    
class RMSNorm(Module):
    def __init__(self, d_model: int, 
                 eps: float = 1e-5, 
                 device: torch.device | None = None, 
                 dtype: torch.dtype | None = None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.gamma = Parameter(torch.tensor([1.0]*d_model, dtype=dtype, device=device))

    def forward(self, h: Tensor) -> Tensor:
        norm = torch.sqrt((h**2).sum(-1)/self.d_model + self.eps).unsqueeze(-1)
        return h / norm * self.gamma

class FFN(Module):
    def __init__(self, d_model: int, 
                 d_ff: int, 
                 device: torch.device | None = None, 
                 dtype: torch.dtype | None = None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.l_1 = Linear(self.d_model, self.d_ff, device, dtype)
        self.l_2 = Linear(self.d_ff, self.d_model, device, dtype)
        self.l_3 = Linear(self.d_model, self.d_ff, device, dtype)

    def forward(self, h: Tensor) -> Tensor:
        return self.l_2(silu(self.l_1(h)) * self.l_3(h))