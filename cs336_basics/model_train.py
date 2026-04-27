import torch
from torch import Tensor
from torch.nn import Module, ModuleList
from torch.nn import Linear, Embedding, RMSNorm
from torch.nn.functional import silu, scaled_dot_product_attention

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
        pos = torch.arange(0, max_seq_len, device=device)[:, None]
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
    

from torch.utils.data import IterableDataset, DataLoader
import tiktoken
from os import PathLike

class Dataset(IterableDataset):
    def __init__(self, path: str | PathLike, tokenizer: tiktoken.Encoding, 
                 block_size: int, buffer_size: int = 10000, 
                 device: torch.device | None = None):
        super().__init__()
        self.path = path
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.buffer_size = buffer_size
        assert self.buffer_size > self.block_size

    def __iter__(self):
        buffer = []

        with open(self.path, 'r', encoding='utf-8') as f:
            while True:
                if len(buffer) < self.buffer_size:
                    chunk = f.read(self.buffer_size)
                    if not chunk:
                        break
                    token_ids = self.tokenizer.encode(chunk, allowed_special='all')
                    buffer.extend(token_ids)
                
                sample = buffer[:self.block_size+1]
                buffer = buffer[self.block_size:]

                yield torch.tensor(sample)


if __name__ == '__main__':
    import pathlib, os
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default=None)
    parser.add_argument('--ckpt', default=None)
    args = parser.parse_args()

    DATA_TRAIN_PATH = pathlib.Path(args.data).resolve() / "TinyStoriesV2-GPT4-train.txt" if args.data else (pathlib.Path(__file__).resolve().parent.parent) / "data" / "TinyStoriesV2-GPT4-train.txt"
    DATA_VAL_PATH = pathlib.Path(args.data).resolve() / "TinyStoriesV2-GPT4-valid.txt" if args.data else (pathlib.Path(__file__).resolve().parent.parent) / "data" / "TinyStoriesV2-GPT4-valid.txt"
    CHECKPOINT_PATH = pathlib.Path(args.ckpt).resolve() / "checkpoint.pt" if args.ckpt else (pathlib.Path(__file__).resolve().parent.parent) / "checkpoint" / 'checkpoint.pt'

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        try:
            import torch_xla
            device = torch_xla.device()
        except:
            device = torch.device('cpu')

    tokenizer = tiktoken.get_encoding('gpt2')

    data_conf = {"context_length": 1024, "batch_size": 6} # 32

    dataset_train = Dataset(DATA_TRAIN_PATH, tokenizer, block_size=data_conf['context_length'])

    dataloader_train = DataLoader(dataset_train, batch_size = data_conf['batch_size'], num_workers=1)

    model_conf = {"vocab_size": tokenizer.n_vocab, "num_layers": 4, 
                  "d_model": 512, "num_heads": 4, "d_ff": 1344,  
                  "theta": 10000., "max_seq_len": 2048,
                  "device": device, 'dtype': torch.float32}

    model = TransformerLM(**model_conf)

    from torch.optim import AdamW
    from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR, ConstantLR
    

    opt_conf = {'lr_max': 1e-3, 'lr_min': 1e-4, 'T_w': 100, 'T_c': 1000, 'T': 1500}

    optimizer = AdamW(model.parameters(), opt_conf['lr_max'])

    scheduler_warm_up = LinearLR(optimizer, start_factor=1e-3, end_factor=1.0, total_iters=opt_conf['T_w'])
    scheduler_cos_ann = CosineAnnealingLR(optimizer, T_max=opt_conf['T_c'] - opt_conf['T_w'], eta_min=opt_conf['lr_min'])
    scheduler_post_ann = ConstantLR(optimizer, factor=1.0, total_iters=opt_conf['T'] - opt_conf['T_c'])

    scheduler = SequentialLR(optimizer=optimizer, 
                             schedulers=[scheduler_warm_up, scheduler_cos_ann, scheduler_post_ann],
                             milestones=[opt_conf['T_w'], opt_conf['T_c']])

    from .checkpointing import save_checkpoint, load_checkpoint

    checkpoint_conf = {'checkpoint': 100, 'logging': 10, 'val': 10}

    if not os.path.exists(CHECKPOINT_PATH.parent):
        os.mkdir(CHECKPOINT_PATH.parent)


    save_checkpoint(model, optimizer, 0, CHECKPOINT_PATH)

    from torch.nn.functional import cross_entropy

    for it, data in enumerate(dataloader_train):
        data: Tensor
        data = data.to(device)
        model.zero_grad()

        input = data[:, :-1]
        target = data[:, 1:]
        logits: Tensor = model(input)


        loss = cross_entropy(logits.view(-1, logits.size(-1)), target.contiguous().view(-1))

        loss.backward()

        optimizer.step()
        scheduler.step()
        
        if it % checkpoint_conf['logging'] == 0:
            print(f"loss: {loss.item():.4f}")
        
        if it != 0 and it % checkpoint_conf['checkpoint'] == 0:
            save_checkpoint(model, optimizer, it, CHECKPOINT_PATH) 
        

        