def get_tokens(word: str, vocab_hash: dict):

    tokens = []

    id = 0
    while id < len(word):
        i = id
        d = vocab_hash
        while i < len(word):
            if word[i] in d:
                d = d[word[i]]
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
        l_temp = [""] + vocab_hash_to_list(vocab_hash[key])
        l_temp = [key + v for v in l_temp]
        l += l_temp
            
    return l
            
def vocab_hash_add(vocab_hash: dict, v: str) -> None:
    d = vocab_hash
    for id in range(len(v)):
        if v[id] in d:
            d = d[v[id]]
        else:
            for id_1 in range(id, len(v)):
                d[v[id_1]] = {}
                d = d[v[id_1]]
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

    vocab_ash = {}
    for word in counts.keys():
        word: str
        for c in word:
            vocab_ash[c] = {}

    merges = []
    vocab_len = len(vocab_hash_to_list(vocab_ash))
    while vocab_len < vocab_size:
        merge_counts = {}
        for word in counts.keys():
            count = counts[word]
            tokens = get_tokens(word, vocab_ash)
            for i in range(len(tokens)-1):
                try:
                    merge_counts[(tokens[i], tokens[i+1])] += count
                except:
                    merge_counts[(tokens[i], tokens[i+1])] = count
        (p_m, count) = max(list(merge_counts.items()), key=lambda x: x[1])

        merges.append(p_m)

        vocab_hash_add(vocab_ash, ''.join(p_m))
        vocab_len += 1
        assert vocab_len == len(vocab_hash_to_list(vocab_ash))

    return vocab_ash, merges

if __name__ == '__main__':
    word = "based"

    vocab = {'a': {'s': {}}, 'b': {}, 'd': {}, 'e': {}, 's': {}}

    assert get_tokens(word, vocab) == ['b', 'as', 'e', 'd']

    assert vocab_hash_to_list(vocab) == ['a', 'as', 'b', 'd', 'e', 's']

    vocab_hash_add(vocab, 'attt')

    assert vocab == {'a': {'s': {}, 't': {'t': {'t': {}}}}, 'b': {}, 'd': {}, 'e': {}, 's': {}}

    word = "based"

    vocab = {'a': {'s': {}}, 'b': {'a': {}}, 'd': {}, 'e': {}, 's': {}}

    assert get_tokens(word, vocab) == ['ba', 's', 'e', 'd']

    import os
    import pathlib
    import regex as re

    input_path, vocab_size, special_tokens = ((pathlib.Path(os.getcwd()).resolve()) / "tests" / "fixtures" / "corpus.en",
                                              120, '<|endoftext|>')
    
    # input_path, vocab_size, special_tokens = ((pathlib.Path(__file__).resolve().parent) / "fixtures",
    #                                           100, '<|endoftext|>')

    assert vocab_hash_to_list(train_bpe(input_path, vocab_size, special_tokens)[0]) == ['i', 'in', 'is', 'r', 're', 'o', 'or', 'n', 'nd', ' ', ' t', ' th', ' the', ' a', ' o', ' ,', ' s', ' .', ' w', ' c', 'c', 'e', 'er', 'm', 't', 's', 'a', 'at', 'd', 'y', 'f', 'u', 'p', 'w', 'h', 'he', 'l', 'b', 'k', 'g', '(', ')', '.', '\n', ',', 'v', 'P', 'O', 'B', 'x', 'N', 'T', 'A', 'L', 'W', 'E', 'D', 'I', 'M', 'S', '!', '?', 'V', '5', '#', '@', '-', 'G', '0', '8', '1', '3', '2', '4', '9', 'F', '&', ';', 'U', 'q', 'C', '6', '/', '7', 'Q', 'X', '®', 'H', 'Y', '\xad', 'j', '*', ':', '©', 'à', 'z', 'J', 'K', 'R', 'ü', 'í', 'á', 'Z', '_', '$', '%', 'ñ', '\x93', '\x94', '=', 'õ', 'ä', '€', 'ö', 'å', '\x97', '™', 'ß', '�', '+']
    

