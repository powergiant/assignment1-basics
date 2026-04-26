from typing import Optional, Callable, Iterable
import torch
from torch import Tensor
from torch.nn import Parameter
import math

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        defaults = {'lr': lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()

        for param_group in self.param_groups:
            lr = param_group['lr']
            with torch.no_grad():
                for param in param_group['params']:
                    if param.grad is None:
                        continue
                    
                    param: Parameter
                    param.copy_(param - lr * param.grad)
        
        return loss
    
class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr = 1e-3, weight_decay = 0.01, betas = (0.9, 0.999), eps = 1e-8):
        defaults = {'lr': lr, 'weight_decay': weight_decay, 'betas': betas, 'eps': eps}
        super().__init__(params, defaults)

    def step(self, closure = None):
        loss = None if closure is None else closure()

        with torch.no_grad():

            for param_group in self.param_groups:
                params = param_group['params']
                lr = param_group['lr']
                weight_decay = param_group['weight_decay'] 
                (beta_1, beta_2) = param_group['betas'] 
                eps =  param_group['eps'] 

                for param in params:
                    param: Parameter

                    state = self.state[param]
                    t = state.get('t', 0)
                    m = state.get('m', torch.zeros_like(param))
                    v = state.get('v', torch.zeros_like(param))

                    t = t + 1

                    g = param.grad
                    if g is None:
                        continue

                    lr_t = lr * math.sqrt(1 - beta_2 ** t) / (1 - beta_1 ** t)

                    m = beta_1 * m + (1 - beta_1) * g
                    v = beta_2 * v + (1 - beta_2) * (g ** 2)

                    param.copy_((1 - lr * weight_decay) * param - lr_t * m / (torch.sqrt(v) + eps))

                    state['t'] = t
                    state['m'] = m
                    state['v'] = v

        return loss

def cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    if it < warmup_iters:
        return it / warmup_iters * max_learning_rate
    elif warmup_iters <= it <= cosine_cycle_iters:
        return min_learning_rate + 1 / 2 * (1 + math.cos(math.pi * ((it - warmup_iters) / (cosine_cycle_iters - warmup_iters)))) * (max_learning_rate - min_learning_rate)
    else:
        return min_learning_rate

def gradient_clipping(params: Iterable[torch.nn.Parameter], max_l2_norm: float, eps: float = 1e-6):
    params = [p for p in params if p.grad is not None]

    total_norm = torch.sqrt(
        sum(torch.sum(p.grad ** 2) for p in params)
    )

    if total_norm > max_l2_norm:
        for param in params:
            param.grad.mul_(max_l2_norm / (total_norm + eps))



if __name__ == '__main__':
    # weights = Parameter(5 * torch.randn((10, 10)))
    # opt = SGD([weights], lr = 1.0)

    # for t in range(10):
    #     opt.zero_grad()
    #     loss = (weights ** 2).mean()
    #     loss.backward()
    #     print(loss.item())
    #     opt.step()

    weights = Parameter(5 * torch.randn((10, 10)))
    opt = AdamW([weights], lr=1e-1)

    for t in range(10):
        opt.zero_grad()
        loss = (weights ** 2).mean()
        loss.backward()
        print(loss.item())
        opt.step()
        
