from typing import Iterable, Iterator
import os
from .bpe_train import pretokenization

def chunk_pretokenization(chunk: str, special_tokens: list[str] | None) -> Iterator[str]:
    words = pretokenization(chunk, special_tokens)
    for word in words:
        yield word

def word_tokenization_list(bs: list[bytes], merges: list[tuple[bytes, bytes]]) -> list[bytes]:
    if len(bs) == 0:
        return bs
    
    b = bs[0]

    if len(bs) > 1:
        b_next = bs[1]
        if (b, b_next) in merges:
            b_merged = b + b_next
            bs = word_tokenization_list([b_merged] + bs[2:], merges)
        else:
            bs = [bs[0]] + word_tokenization_list(bs[1:], merges)
        return bs
    else:
        return bs
            

def word_tokenization(word: str, vocab_inv: dict[bytes, int], merges: dict[bytes, bytes]) -> Iterator[int]:
    bs = [bytes([b]) for b in word.encode('utf-8')]

    bs_step = word_tokenization_list([bytes([b]) for b in word.encode('utf-8')], merges)

    while bs != bs_step:
        bs = bs_step
        bs_step = word_tokenization_list(bs, merges)

    for b in bs:
        yield vocab_inv[b]



class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], 
                 merges: list[tuple[bytes, bytes]], 
                 special_tokens: list[str]=None):
        self.vocab = vocab
        self.vocab_inv = {b: id for id, b in vocab.items()}
        self.merges = merges
        self.special_tokens = special_tokens

    def from_files(cls, vocab_filepath: str | os.PathLike, merges_filepath: str | os.PathLike, special_tokens: list[str] | None = None):
        pass

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for chunk in iterable:
            words = chunk_pretokenization(chunk, self.special_tokens)
            for word in words:
                tokens = word_tokenization(word, self.vocab_inv, self.merges)
                for token in tokens:
                    yield token


    def decode(self, ids: list[int]) -> str:
        s = b""
        for id in ids:
            s += self.vocab[id]
        return s.decode('utf-8')

def test_chunk_pretokenization():
    assert [word for word in chunk_pretokenization('acass acas', None)] == ['acass', ' acas']

def test_word_tokenization_list(merges: list[tuple[bytes, bytes]]):
    assert word_tokenization_list([b'a', b'c', b'a', b's', b's', b' '], merges) == [b'a', b'c', b'as', b's', b' ']
    assert word_tokenization_list([b'a', b'c', b'a', b's', b's', b'a'], merges) == [b'a', b'c', b'as', b'sa']

def test_word_tokenization(tokenizer: Tokenizer):
    assert list(word_tokenization('acassa', tokenizer.vocab_inv, tokenizer.merges)) == [1, 2, 6]

def test_encode_iterable(tokenizer: Tokenizer):
    print(list(tokenizer.encode_iterable(['acassa acass'])))
    assert list(tokenizer.encode_iterable(['acassa acass'])) == [1, 2, 6, 7, 1, 2, 4, 3]


def test_decode(tokenizer):
    assert tokenizer.decode([1, 2, 4, 3]) == 'acass'
    

if __name__ == '__main__':
    vocab = {1: b'a', 2: b'c', 3: b's', 4: b'as', 5: b'sa', 6: b'assa', 7: b' '}
    merges = [(b'a', b's'), (b's', b'a'), (b'as', b'sa')]
    tokenizer = Tokenizer(vocab, merges)

    test_decode(tokenizer)
    test_chunk_pretokenization()
    test_word_tokenization_list(merges)
    test_word_tokenization(tokenizer)
    test_encode_iterable(tokenizer)