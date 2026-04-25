import regex as re


# def replace_pair(l: list[bytes], p_m: tuple[bytes, bytes]):
#     id_s = 0
#     id_e = 0
#     n = len(l)
#     l_new = []
#     while id_e < n:
#         if id_e < n - 1 and l[id_e] == p_m[0] and l[id_e+1] == p_m[1]:
#             l_new.extend(l[id_s:id_e])
#             l_new.append(p_m[0] + p_m[1])
#             id_e += 2
#             id_s = id_e
#         else:
#             id_e += 1
#     l_new.extend(l[id_s:id_e])
#     return l_new

def replace_pair(l: list[bytes], p_m: tuple[bytes, bytes]):
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

def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    TODO:
    """
    chunk = ""
    with open(input_path, 'r') as f:
        while True:
            line = f.readline()
            if len(line) != 0:
                chunk += line
            else:
                break

    pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    words = re.finditer(pattern, chunk)

    counts = {}
    for word in words:
        try:
            counts[word.group()] += 1
        except:
            counts[word.group()] = 1

    vocab = [(bytes([i]), i) for i in range(256)]

    merges = []
    vocab_len = len(vocab)
    counts: dict[str, int]
    words_tokens = {word: [bytes([c]) for c in word.encode("utf-8")] for word in counts.keys()}
    while vocab_len < vocab_size:
        merge_counts = {}
        for word in counts.keys():
            count = counts[word]
            tokens = words_tokens[word]
            for i in range(len(tokens)-1):
                try:
                    merge_counts[(tokens[i], tokens[i+1])] += count
                except:
                    merge_counts[(tokens[i], tokens[i+1])] = count
        (p_m, count) = max(list(merge_counts.items()), key=lambda x: x[1])

        for word in words_tokens.keys():
            words_tokens[word] = replace_pair(words_tokens[word], p_m)

        merges.append(p_m)

        vocab.append((p_m[0]+p_m[1], vocab_len))
        vocab_len += 1

    return vocab, merges

if __name__ == '__main__':

    import os
    import pathlib
    import time

    input_path, vocab_size, special_tokens = ((pathlib.Path(os.getcwd()).resolve()) / "tests" / "fixtures" / "corpus.en",
                                              500, '<|endoftext|>')
    
    # input_path, vocab_size, special_tokens = ((pathlib.Path(__file__).resolve().parent) / "fixtures",
    #                                           100, '<|endoftext|>')

    start_time = time.time()
    vocab = train_bpe(input_path, vocab_size, special_tokens)[0][272:292]
    print(vocab)
    assert vocab == [(b' c', 272), (b'on', 273), (b' b', 274), (b' f', 275), (b'en', 276), (b'ou', 277), (b'it', 278), (b'es', 279), (b' of', 280), (b' p', 281), (b'ed', 282), (b'ing', 283), (b' in', 284), (b'al', 285), (b' m', 286), (b' and', 287), (b' d', 288), (b'an', 289), (b'ar', 290), (b' to', 291)]
    end_time = time.time()
    print(end_time - start_time)
    assert end_time - start_time < 1.5
    

