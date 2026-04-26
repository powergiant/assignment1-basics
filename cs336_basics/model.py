import torch
from torch import Tensor
from torch.nn import Module, Parameter
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
        # return einsum(in_features, self.weight, '... in_feature, ... out_feature in_feature -> ... out_feature')

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

def silu(x: Tensor) -> Tensor:
    return x/(1+torch.exp(-x))

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
    
class RoPE(Module):
    def __init__(self, theta: float, d_model: int, 
                 max_seq_len: int, 
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None):
        super().__init__()
        self.theta = theta
        self.d_model = d_model
        assert d_model % 2 == 0
        self.max_seq_len = max_seq_len
        self.freq: Tensor
        theta = torch.tensor(self.theta, device=device, dtype=dtype)
        # freq = theta ** torch.tensor([[-k/(d_model//2) for k in range(d_model//2)]], device=device, dtype=dtype)
        freq = theta ** (- torch.arange(0, d_model//2, device=device, dtype=dtype) / (d_model//2))
        self.register_buffer('freq', freq, persistent=False)
    
    def forward(self, h: Tensor, pos: Tensor) -> Tensor:
        assert h.size(-1) == self.d_model
        assert h.size(-2) == pos.size(-1)
        shape = h.shape
        h = h.reshape(*shape[:-1], -1, 2)
        pos = pos.unsqueeze(-1)
        cos = torch.cos(pos * self.freq)
        sin = torch.sin(pos * self.freq)
        return torch.stack((cos * h[..., 0] - sin * h[..., 1], 
                          sin * h[..., 0] + cos * h[..., 1]), dim=-1).reshape(shape)

def softmax(h: Tensor, d: int | None = None) -> Tensor:
    if d is not None:
        h = h - torch.max(h, dim=d).values.unsqueeze(-1).detach()
    else:
        h = h - torch.max(h, dim=-1).values.unsqueeze(-1).detach()
    exp = torch.exp(h)
    norm= exp.sum(-1).unsqueeze(-1)
    return exp/norm

def scaled_dot_product_attention(Q: Tensor, K: Tensor, V: Tensor, mask: Tensor) -> Tensor:
    d = Q.size(-1)
    scores = Q @ K.transpose(-1, -2)/math.sqrt(d)
    scores = scores.masked_fill(~mask, - torch.inf)
    return softmax(scores) @ V

class MultiheadSelfAttentionWithouRoPE(Module):
    def __init__(self, d_model: int, num_heads: int,
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        assert self.d_model % (self.num_heads * 2) == 0
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.o_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.device = device
    
    def forward(self, h: Tensor) -> Tensor:
        assert h.shape[-1] == self.d_model
        B, T, D, H = (h.shape[:-2], h.shape[-2], self.d_model, self.num_heads)
        Q = self.q_proj.forward(h)
        K = self.k_proj.forward(h)
        V = self.v_proj.forward(h)
        
        Q = Q.view(*B, T, H, D//H).transpose(-2, -3)
        K = K.view(*B, T, H, D//H).transpose(-2, -3)
        V = V.view(*B, T, H, D//H).transpose(-2, -3)

        mask = torch.tril(torch.ones(T, T, dtype=torch.bool), diagonal=0)

        h = scaled_dot_product_attention(Q, K, V, mask).transpose(-2, -3).contiguous()

        return self.o_proj(h.view(*B, T, D))

class MultiheadSelfAttention(Module):
    def __init__(self, d_model: int, num_heads: int, theta: float, max_seq_len: int,
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        assert self.d_model % (self.num_heads * 2) == 0
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.o_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.rope = RoPE(theta, d_model//self.num_heads, max_seq_len, device=device, dtype=dtype)
        self.device = device
    
    def forward(self, h: Tensor) -> Tensor:
        assert h.shape[-1] == self.d_model
        B, T, D, H = (h.shape[:-2], h.shape[-2], self.d_model, self.num_heads)
        Q = self.q_proj.forward(h)
        K = self.k_proj.forward(h)
        V = self.v_proj.forward(h)
        
        Q = Q.view(*B, T, H, D//H).transpose(-2, -3)
        K = K.view(*B, T, H, D//H).transpose(-2, -3)
        V = V.view(*B, T, H, D//H).transpose(-2, -3)

        pos = torch.arange(0, T, device=self.device)

        Q_rotate = self.rope.forward(Q, pos)
        K_rotate = self.rope.forward(K, pos)

        mask = torch.tril(torch.ones(T, T, dtype=torch.bool), diagonal=0)

        h = scaled_dot_product_attention(Q_rotate, K_rotate, V, mask).transpose(-2, -3).contiguous()

        return self.o_proj(h.view(*B, T, D))




        

if __name__ == '__main__':
    x = torch.tensor([[1.0, 3.0, 1.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    from torch.nn.functional import softmax as softmax_g, scaled_dot_product_attention as scaled_dot_product_attention_g

    print(softmax_g(x) - softmax(x))
    print(softmax(x))

    Q = x
    K = x
    V = x
    mask = x > 2.1
    print(mask)
    print(scaled_dot_product_attention_g(Q, K, V, mask))
    print(scaled_dot_product_attention(Q, K, V, mask))

    # print(torch.ones(5, 5, dtype=torch.bool))
    # print(torch.tril(torch.ones(5, 5, dtype=torch.bool), diagonal=0))


