import regex as re
from typing import BinaryIO
from regex import Scanner

def replace_pair(l: list[bytes], p_m: tuple[bytes, bytes]):
    if p_m[0] not in l:
        return l
    id = 0
    n = len(l)
    l_new = []
    while id < n:
        if id < n - 1 and l[id] == p_m[0] and l[id+1] == p_m[1]:
            l_new.append(p_m[0] + p_m[1])
            id += 2
        else:
            l_new.append(l[id])
            id += 1
    return l_new

def pretokenization(input_path: str, special_tokens: list[str]) -> list[str]:
    chunk = ""
    with open(input_path, 'r') as f:
        while True:
            line = f.readline()
            if len(line) != 0:
                chunk += line
            else:
                break

    pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    words = [word for word in re.finditer(pattern, chunk)]

    return words



def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    TODO:
    """
    words = pretokenization(input_path, special_tokens)
    counts = {}
    for word in words:
        try:
            counts[word.group()] += 1
        except:
            counts[word.group()] = 1

    vocab = {i: bytes([i]) for i in range(256)}

    merges = []
    vocab_len = len(vocab)
    counts: dict[str, int]
    words_tokens = {word: [bytes([c]) for c in word.encode("utf-8")] for word in counts.keys()}
    while vocab_len < vocab_size - len(special_tokens):
        merge_counts = {}
        for word in counts.keys():
            count = counts[word]
            tokens = words_tokens[word]
            for i in range(len(tokens)-1):
                try:
                    merge_counts[(tokens[i], tokens[i+1])] += count
                except:
                    merge_counts[(tokens[i], tokens[i+1])] = count
        (p_m, count) = max(list(merge_counts.items()), key=lambda x: (x[1], x[0]))

        for word in words_tokens.keys():
            words_tokens[word] = replace_pair(words_tokens[word], p_m)

        merges.append(p_m)

        vocab[vocab_len] = p_m[0]+p_m[1]
        vocab_len += 1

    for id, special_token in enumerate(special_tokens):
        vocab[vocab_size - len(special_tokens) + id] = bytes(special_token.encode("utf-8"))

    return vocab, merges

def test_corpus_en():
    import os
    import pathlib
    import time

    input_path, vocab_size, special_tokens = ((pathlib.Path(os.getcwd()).resolve()) / "tests" / "fixtures" / "corpus.en",
                                              500, '<|endoftext|>')
    
    # input_path, vocab_size, special_tokens = ((pathlib.Path(__file__).resolve().parent) / "fixtures",
    #                                           100, '<|endoftext|>')

    start_time = time.time()
    vocab = list(train_bpe(input_path, vocab_size, special_tokens)[0].items())[272:292]
    print(vocab)
    assert vocab == [(272, b' c'), (273, b'on'), (274, b' b'), (275, b' f'), (276, b'ou'), (277, b'it'), (278, b'en'), (279, b'es'), (280, b' of'), (281, b' p'), (282, b'ing'), (283, b' in'), (284, b'ed'), (285, b'al'), (286, b' m'), (287, b' and'), (288, b' d'), (289, b'an'), (290, b'ar'), (291, b' to')]
    end_time = time.time()
    print(end_time - start_time)
    assert end_time - start_time < 1.5

def test_tinystories():
    import os
    import pathlib
    import time

    input_path, vocab_size, special_tokens = ((pathlib.Path(os.getcwd()).resolve()) / "tests" / "fixtures" / "tinystories_sample_5M.txt",
                                              500, '<|endoftext|>')
    
    # input_path, vocab_size, special_tokens = ((pathlib.Path(__file__).resolve().parent) / "fixtures",
    #                                           100, '<|endoftext|>')

    start_time = time.time()
    vocab = list(train_bpe(input_path, vocab_size, special_tokens)[0].items())[272:292]
    print(vocab)
    assert vocab == [(272, b' c'), (273, b'on'), (274, b' b'), (275, b' f'), (276, b'ou'), (277, b'it'), (278, b'en'), (279, b'es'), (280, b' of'), (281, b' p'), (282, b'ing'), (283, b' in'), (284, b'ed'), (285, b'al'), (286, b' m'), (287, b' and'), (288, b' d'), (289, b'an'), (290, b'ar'), (291, b' to')]
    end_time = time.time()
    print(end_time - start_time)
    assert end_time - start_time < 1.5


if __name__ == '__main__':

    test_corpus_en()