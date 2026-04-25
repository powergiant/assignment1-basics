@profile
def get_tokens(word: str, vocab_hash: dict) -> bytes:

    word = bytes(word.encode('utf-8'))

    tokens = []

    id = 0
    while id < len(word):
        i = id
        d = vocab_hash
        while i < len(word):
            b_i = bytes([word[i]])
            if b_i in d:
                d = d[b_i]
                i += 1
            else:
                break
        token = word[id:i]
        id = i
        tokens.append(token)
    
    return tokens

def vocab_hash_to_list(vocab_hash: dict) -> list:
    l = []

    if vocab_hash == {}:
        return []
    
    for key in vocab_hash.keys():
        l_temp = [b""] + vocab_hash_to_list(vocab_hash[key])
        l_temp = [key + v for v in l_temp]
        l += l_temp
            
    return l
            
def vocab_hash_add(vocab_hash: dict, v: str) -> None:
    d = vocab_hash
    for id in range(len(v)):
        b_id = bytes([v[id]])
        if b_id in d:
            d = d[b_id]
        else:
            for id_1 in range(id, len(v)):
                b_id_1 = bytes([v[id_1]])
                d[b_id_1] = {}
                d = d[b_id_1]
            break

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

    vocab_hash = {bytes([i]): {} for i in range(256)}
    # for word in counts.keys():
    #     word: str
    #     for c in word.encode('utf-8'):
    #         vocab_hash[bytes([c])] = {}
    vocab = [(key, key[0]) for key in vocab_hash.keys()]

    merges = []
    vocab_len = len(vocab)
    while vocab_len < vocab_size:
        merge_counts = {}
        for word in counts.keys():
            count = counts[word]
            tokens = get_tokens(word, vocab_hash)
            for i in range(len(tokens)-1):
                try:
                    merge_counts[(tokens[i], tokens[i+1])] += count
                except:
                    merge_counts[(tokens[i], tokens[i+1])] = count
        (p_m, count) = max(list(merge_counts.items()), key=lambda x: x[1])

        merges.append(p_m)

        vocab_hash_add(vocab_hash, p_m[0]+p_m[1])
        vocab.append((p_m[0]+p_m[1], vocab_len))
        vocab_len += 1

    return vocab, merges

if __name__ == '__main__':
    word = "based"

    vocab_hash = {b'a': {b's': {}}, b'b': {}, b'd': {}, b'e': {}, b's': {}}

    assert get_tokens(word, vocab_hash) == [b'b', b'as', b'e', b'd']

    assert vocab_hash_to_list(vocab_hash) == [b'a', b'as', b'b', b'd', b'e', b's']

    vocab_hash_add(vocab_hash, b'attt')

    assert vocab_hash == {b'a': {b's': {}, b't': {b't': {b't': {}}}}, b'b': {}, b'd': {}, b'e': {}, b's': {}}

    # word = "based"

    # vocab = {'a': {'s': {}}, 'b': {'a': {}}, 'd': {}, 'e': {}, 's': {}}

    # assert get_tokens(word, vocab) == ['ba', 's', 'e', 'd']

    import os
    import pathlib
    import regex as re
    import time

    input_path, vocab_size, special_tokens = ((pathlib.Path(os.getcwd()).resolve()) / "tests" / "fixtures" / "corpus.en",
                                              500, '<|endoftext|>')
    
    # input_path, vocab_size, special_tokens = ((pathlib.Path(__file__).resolve().parent) / "fixtures",
    #                                           100, '<|endoftext|>')

    start_time = time.time()
    assert train_bpe(input_path, vocab_size, special_tokens)[0][272:292] == [(b' c', 272), (b'on', 273), (b' b', 274), (b' f', 275), (b'en', 276), (b'ou', 277), (b'it', 278), (b'es', 279), (b' of', 280), (b' p', 281), (b'ed', 282), (b'ing', 283), (b' in', 284), (b'al', 285), (b' m', 286), (b' and', 287), (b' d', 288), (b'an', 289), (b'ar', 290), (b' to', 291)]
    end_time = time.time()
    print(end_time - start_time)
    assert end_time - start_time < 1.5
    

