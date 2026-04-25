from typing import Iterable, Iterator
import os
from .bpe_train import pretokenization, replace_pair

pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def chunk_pretokenization_iter(chunk: str, special_tokens: list[str] | None) -> Iterator[str]:
    words = pretokenization(chunk, special_tokens, pattern)
    for word in words:
        yield word

def chunk_pretokenization_list(chunk: str, special_tokens: list[str] | None) -> list[str]:
    return pretokenization(chunk, special_tokens, pattern)

def word_tokenization_list_rec(bs: list[bytes], merges: list[tuple[bytes, bytes]]) -> list[bytes]:
    is_there_merge = False
    for b_0, b_1 in merges:
        if b_0 not in bs:
            continue
        else:
            for id in range(len(bs)-1):
                if b_0 == bs[id] and b_1 == bs[id+1]:
                    bs = replace_pair(bs, (b_0, b_1))
                    is_there_merge = True
                    break
    if is_there_merge:
        return word_tokenization_list_rec(bs, merges)
    else:
        return bs
    

def word_tokenization_list(word: str, vocab_inv: dict[bytes, int], merges: list[tuple[bytes, bytes]]) -> list[int]:
    bs = [bytes([b]) for b in word.encode('utf-8')]

    bs = word_tokenization_list_rec(bs, merges)

    return [vocab_inv[b] for b in bs]


def word_tokenization_iter(word: str, vocab_inv: dict[bytes, int], merges: list[tuple[bytes, bytes]]) -> Iterator[int]:
    ids = word_tokenization_list(word, vocab_inv, merges)

    for id in ids:
        yield id



class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], 
                 merges: list[tuple[bytes, bytes]], 
                 special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.vocab_inv = {b: id for id, b in vocab.items()}
        self.merges = merges
        self.special_tokens = special_tokens

    def from_files(cls, vocab_filepath: str | os.PathLike, merges_filepath: str | os.PathLike, special_tokens: list[str] | None = None):
        pass

    def encode(self, text: str) -> list[int]:
        l_enc = []
        words = chunk_pretokenization_list(text, self.special_tokens)
        for word in words:
            if self.special_tokens and word in self.special_tokens:
                l_enc.append(self.vocab_inv[word.encode('utf-8')])
            else:
                l_enc += word_tokenization_list(word, self.vocab_inv, self.merges)
        return l_enc
    

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for chunk in iterable:
            words = chunk_pretokenization_iter(chunk, self.special_tokens)
            for word in words:
                if self.special_tokens and word in self.special_tokens:
                    tokens = [self.vocab_inv[word.encode('utf-8')]]
                else:
                    tokens = word_tokenization_iter(word, self.vocab_inv, self.merges)
                for token in tokens:
                    yield token


    def decode(self, ids: list[int]) -> str | None:
        s = b""
        for id in ids:
            s += self.vocab[id]
        try:
            return s.decode('utf-8')
        except:
            return None

def test_chunk_pretokenization():
    assert [word for word in chunk_pretokenization_iter('acass acas', None)] == ['acass', ' acas']

def test_word_tokenization_list_rec(merges: list[tuple[bytes, bytes]]):
    assert word_tokenization_list_rec([b'a', b'c', b'a', b's', b's', b' '], merges) == [b'a', b'c', b'as', b's', b' ']
    assert word_tokenization_list_rec([b'a', b'c', b'a', b's', b's', b'a'], merges) == [b'a', b'c', b'assa']

def test_word_tokenization(tokenizer: Tokenizer):
    assert list(word_tokenization_iter('acassa', tokenizer.vocab_inv, tokenizer.merges)) == [1, 2, 6]

def test_encode_iterable(tokenizer: Tokenizer):
    assert list(tokenizer.encode_iterable(['acassa acass'])) == [1, 2, 6, 7, 1, 2, 4, 3]

def test_encode(tokenizer: Tokenizer):
    assert list(tokenizer.encode('')) == []
    assert list(tokenizer.encode('acassa acass')) == [1, 2, 6, 7, 1, 2, 4, 3]


def test_decode(tokenizer):
    assert tokenizer.decode([1, 2, 4, 3]) == 'acass'
    

if __name__ == '__main__':
    vocab = {1: b'a', 2: b'c', 3: b's', 4: b'as', 5: b'sa', 6: b'assa', 7: b' '}
    merges = [(b'a', b's'), (b's', b'a'), (b'as', b'sa')]
    tokenizer = Tokenizer(vocab, merges)

    test_decode(tokenizer)
    test_chunk_pretokenization()
    test_word_tokenization_list_rec(merges)
    test_word_tokenization(tokenizer)
    test_encode_iterable(tokenizer)
    test_encode(tokenizer)

    vocab = {1: b'h', 2: b'e', 3: b'l', 4: b'l', 5: b'o', 6: b'he', 7: b'll', 8: b'hell', 9: b'llo', 10: b'hello'}
    merges = [(b'h', b'e'), (b'l', b'l'), (b'he', b'll'), (b'll', b'o'), (b'hell', b'o')]
    tokenizer = Tokenizer(vocab, merges)

    assert list(tokenizer.encode('hello')) == [10]