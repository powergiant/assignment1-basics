from .model import TransformerLM, softmax
from .tokenizer import Tokenizer
from collections.abc import Iterator
import torch

def next_id(token_ids: list[int], model: TransformerLM, temp: float, top_p: float | None = None) -> int:
    with torch.no_grad():
        logits = model.forward(token_ids)
        prob = softmax(logits/temp)
        if top_p is None:
            token_next_id = torch.multinomial(prob, num_samples=1).item()
        else:
            prob_sort, id_sorted = prob.sort(-1, descending=True)
            keep = prob_sort.cumsum(dim=-1) <= top_p
            keep[..., 1:] = keep[..., :-1].clone()
            keep[..., 0] = True
            ids_sampled = id_sorted[keep]
            prob_sampled = prob_sort[keep]
            token_next_id = ids_sampled[torch.multinomial(prob_sampled, num_samples=1)].item()
        return token_next_id

def generate(prompt: str, model: TransformerLM, tokenizer: Tokenizer, temp: float, top_p: float | None = None) -> Iterator[str]:
    token_ids = tokenizer.encode(prompt)
    buffer = []
    while True:
        id = next_id(token_ids, model, temp, top_p)
        token_ids.append(id)
        buffer.append(id)
        word = tokenizer.decode(buffer) 
        if word is not None:
            yield word
            buffer.clear()
        else:
            continue

# def generate(token_ids: list[int], model: TransformerLM, temp: float, top_p: float | None = None) -> Iterator[int]:
#     token_ids_inner = token_ids.copy()
#     while True:
#         token_id = generate(token_ids_inner, model, temp, top_p)
#         token_ids_inner.append(token_id)
#         yield token_id

end_token: str = '<|endoftext|>'
    