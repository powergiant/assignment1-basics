import torch
from torch import Tensor
from torch.nn import Module, ModuleList
from torch.nn import Linear, Embedding, RMSNorm
from torch.nn.functional import silu, scaled_dot_product_attention
from torch.nn.attention import sdpa_kernel, SDPBackend

class FFN(Module):
    def __init__(self, d_model: int, 
                 d_ff: int, 
                 device: torch.device | None = None, 
                 dtype: torch.dtype | None = None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = Linear(self.d_model, self.d_ff, bias=False, device=device, dtype=dtype)
        self.w2 = Linear(self.d_ff, self.d_model, bias=False, device=device, dtype=dtype)
        self.w3 = Linear(self.d_model, self.d_ff, bias=False, device=device, dtype=dtype)

    def forward(self, h: Tensor) -> Tensor:
        return self.w2(silu(self.w1(h)) * self.w3(h))
    
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
        
        self.cos: Tensor
        self.sin: Tensor
        theta = torch.tensor(self.theta, device=device, dtype=dtype)
        freq = (theta ** (- torch.arange(0, d_model//2, device=device, dtype=dtype) / (d_model//2)))
        pos = torch.arange(0, max_seq_len)[:, None]
        cos = torch.cos(pos * freq)
        sin = torch.sin(pos * freq)
        
        self.register_buffer('cos', cos, persistent=False)
        self.register_buffer('sin', sin, persistent=False)

    def forward(self, h: Tensor, pos: Tensor) -> Tensor:
        assert h.size(-1) == self.d_model
        assert h.size(-2) == pos.size(-1)
        shape = h.shape
        h = h.reshape(*shape[:-1], -1, 2)
        cos = self.cos[pos]
        sin = self.sin[pos]
        return torch.stack((cos * h[..., 0] - sin * h[..., 1], 
                          sin * h[..., 0] + cos * h[..., 1]), dim=-1).reshape(shape)

class MultiheadSelfAttention(Module):
    def __init__(self, d_model: int, num_heads: int, theta: float, max_seq_len: int,
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        assert self.d_model % (self.num_heads * 2) == 0
        self.q_proj = Linear(d_model, d_model, bias=False, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, bias=False, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, bias=False, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, bias=False, device=device, dtype=dtype)
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

        # mask = torch.tril(torch.ones(T, T, dtype=torch.bool), diagonal=0)
        # h = scaled_dot_product_attention(Q_rotate, K_rotate, V, mask).transpose(-2, -3).contiguous()

        h = scaled_dot_product_attention(Q_rotate, K_rotate, V, is_causal=True).transpose(-2, -3).contiguous()

        return self.output_proj(h.view(*B, T, D))


class TransformerBlock(Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, theta: float, max_seq_len: int,
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None):
        super().__init__()
        self.attn = MultiheadSelfAttention(d_model=d_model, 
                                           num_heads=num_heads, 
                                           theta=theta, 
                                           max_seq_len=max_seq_len, 
                                           device=device, 
                                           dtype=dtype)
        self.ln1 = RMSNorm(d_model, eps=1e-5, device=device, dtype=dtype)
        self.ffn = FFN(d_model=d_model, d_ff=d_ff, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, eps=1e-5, device=device, dtype=dtype)

    def forward(self, h: Tensor) -> Tensor:
        h = h + self.attn(self.ln1(h))
        return h + self.ffn(self.ln2(h))
        


class TransformerLM(Module):
    def __init__(self, vocab_size: int, num_layers: int, d_model: int, 
                 num_heads: int, d_ff: int, 
                 theta: float, max_seq_len: int,
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None):
        super().__init__()
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = ModuleList([TransformerBlock(d_model=d_model, num_heads=num_heads, d_ff=d_ff,
                                              theta=theta, max_seq_len=max_seq_len, device=device,
                                              dtype=dtype) for _ in range(num_layers)])
        self.ln_final = RMSNorm(d_model, eps=1e-5, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, bias=False, device=device, dtype=dtype)
    
    def forward(self, token_ids: Tensor) -> Tensor:
        h = self.token_embeddings(token_ids)
        for layer in self.layers:
            h = layer(h)
        logits = self.lm_head(self.ln_final(h))
        return logits


