from typing import Iterable, Iterator
import os
from .bpe_train import pretokenization

def chunk_pretokenization(chunk: str, special_tokens: list[str]) -> Iterator[str]:
    words = pretokenization(chunk, special_tokens)
    for word in words:
        yield word

def word_tokenization_list(tokens: list[int], vocab: dict[int, bytes], merges: dict[bytes, bytes]) -> list[int]:
    token = vocab[tokens[0]]

    if len(tokens) > 1:
        token_next = vocab[tokens[1]]
        if token_next == merges[token]:
            tokens = word_tokenization_list([vocab[bytes(tokens[0:2])]] + tokens[2:], vocab, merges)
        else:
            tokens = [tokens[0]] + word_tokenization_list(tokens[2:], vocab, merges)
        return word_tokenization_list(tokens)
    else:
        return tokens
            

def word_tokenization(word: str, vocab: dict[int, bytes], merges: dict[bytes, bytes]) -> Iterator[int]:
    tokens = word_tokenization_list([c for c in word.encode('utf-8')])

    for token in tokens:
        yield token
    
    # tokens = [c for c in word.encode('utf-8')]

    # token = vocab[tokens[0]]


    # id = 0

    # while id < len(tokens):
    #     id = 0
    #     while id < len(tokens):
    #         token = vocab[tokens[id]]
    #         id_1 = id + 1
    #         while True:
    #             if token not in merges:
    #                 break
    #             else:
    #                 if id_1 < len(tokens):
    #                     token_next = vocab[tokens[id_1]]
    #                     if token_next == merges[token]:
    #                         id_1 += 1
    #                         token += token_next
    #         tokens = [vocab[tokens[id_1]]] + tokens[id_1+1:]
    #         id += 1
    #     id += 1



class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], 
                 merges: list[tuple[bytes, bytes]], 
                 special_tokens: list[str]=None):
        self.vocab = vocab
        self.merges = {token_1: token_2 for token_1, token_2 in merges}
        self.special_tokens = special_tokens

    def from_files(cls, vocab_filepath: str | os.PathLike, merges_filepath: str | os.PathLike, special_tokens: list[str] | None = None):
        pass

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for chunk in iterable:
            words = chunk_pretokenization(chunk, self.special_tokens)
            for word in words:
                tokens = word_tokenization(word)
                for token in tokens:
                    yield token


    def decode(self, ids: list[int]) -> str:
        pass

def test_chunk_pretokenization():
    pass

if __name__ == '__main__':
    pass