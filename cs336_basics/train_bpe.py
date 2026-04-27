import regex as re

def replace_pair(l: list[bytes], p_m: tuple[bytes, bytes]):
    if p_m[0] not in l or p_m[1] not in l:
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

def dict_update(d: dict, key, val: int):
    try:
        d[key] += val
    except:
        d[key] = val

def pretokenization(content: str, special_tokens: list[str] | None, pattern: str) -> list[str]:

    content_list = [content]
    
    if special_tokens:

        special_tokens = sorted(special_tokens, key=lambda x: len(x), reverse=True)

        for special_token in special_tokens:
            l_temp = []
            for item in content_list:
                if item in special_tokens:
                    l_temp.append(item)
                    continue
                item_split = item.split(special_token)
                for it in item_split:
                    l_temp.extend([it, special_token])
                l_temp.pop()
            content_list = l_temp

    words = []

    for item in content_list:
        if special_tokens and item in special_tokens:
            words.append(item)
        else:
            words += [word.group() for word in re.finditer(pattern, item)]

    return words

def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    TODO:
    """
    with open(input_path, 'r') as f:
        content = f.read()
    pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    words = pretokenization(content, special_tokens, pattern)
    words = list(filter(lambda x : x not in special_tokens, words))
    word_counts = {}
    for word in words:
        dict_update(word_counts, word, 1)

    vocab = {i: bytes([i]) for i in range(256)}

    merges = []
    vocab_len = len(vocab)
    word_counts: dict[str, int]
    word_tokens = {word: [bytes([c]) for c in word.encode("utf-8")] for word in word_counts.keys()}
    
    pair_counts = {}
    for word in word_counts.keys():
        count = word_counts[word]
        tokens = word_tokens[word]
        for i in range(len(tokens)-1):
            dict_update(pair_counts, (tokens[i], tokens[i+1]), count)

    while vocab_len < vocab_size - len(special_tokens):
        p_m, _ = max(pair_counts.items(), key=lambda x: (x[1], x[0]))
        p_merged = p_m[0] + p_m[1]
        pair_counts.pop(p_m)
        for word, count in word_counts.items():
            tokens = word_tokens[word]
            if p_m[0] not in tokens or p_m[1] not in tokens:
                continue
            else:
                id = 0
                n = len(tokens)
                tokens_new = []
                while id < n:
                    if id < n - 1 and tokens[id] == p_m[0] and tokens[id+1] == p_m[1]:
                        tokens_new.append(p_merged)
                        if id >= 1:
                            dict_update(pair_counts, (tokens[id-1], tokens[id]), -count)
                            dict_update(pair_counts, (tokens[id-1], p_merged), count)
                        if id < len(tokens) - 2:
                            dict_update(pair_counts, (tokens[id+1], tokens[id+2]), -count)
                            dict_update(pair_counts, (p_merged, tokens[id+2]), count)
                        id += 2
                    else:
                        tokens_new.append(tokens[id])
                        id += 1
                word_tokens[word] = tokens_new

        merges.append(p_m)

        vocab[vocab_len] = p_m[0]+p_m[1]
        vocab_len += 1

    for id, special_token in enumerate(special_tokens):
        vocab[vocab_size - len(special_tokens) + id] = special_token.encode("utf-8")

    return vocab, merges

def test_corpus_en():
    import os
    import pathlib
    import time

    input_path, vocab_size, special_tokens = ((pathlib.Path(os.getcwd()).resolve()) / "tests" / "fixtures" / "corpus.en",
                                              500, ['<|endoftext|>'])
    
    # input_path, vocab_size, special_tokens = ((pathlib.Path(__file__).resolve().parent) / "fixtures",
    #                                           100, ['<|endoftext|>'])

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
                                              500, ['<|endoftext|>'])
    
    # input_path, vocab_size, special_tokens = ((pathlib.Path(__file__).resolve().parent) / "fixtures",
    #                                           100, ['<|endoftext|>'])

    start_time = time.time()
    vocab = list(train_bpe(input_path, vocab_size, special_tokens)[0].items())[272:292]
    print(vocab)
    assert vocab == [(272, b' c'), (273, b'on'), (274, b' b'), (275, b' f'), (276, b'ou'), (277, b'it'), (278, b'en'), (279, b'es'), (280, b' of'), (281, b' p'), (282, b'ing'), (283, b' in'), (284, b'ed'), (285, b'al'), (286, b' m'), (287, b' and'), (288, b' d'), (289, b'an'), (290, b'ar'), (291, b' to')]
    end_time = time.time()
    print(end_time - start_time)
    assert end_time - start_time < 1.5


if __name__ == '__main__':

    test_corpus_en()