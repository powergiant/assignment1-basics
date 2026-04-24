def get_tokens(word: str, vocab: dict):

    tokens = []

    id = 0
    while id < len(word):
        i = id
        d = vocab
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

def vocab_list(vocab: dict) -> list:
    l = []

    if vocab == {}:
        return []
    
    for key in vocab.keys():
        l_temp = [""] + vocab_list(vocab[key])
        l_temp = [key + v for v in l_temp]
        l += l_temp
            
    return l
            
def vocab_add(vocab: dict, v: str) -> None:
    d = vocab
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

    vocab = {}
    for word in counts.keys():
        word: str
        for c in word:
            vocab[c] = {}

    merges = []
    while len(vocab_list(vocab)) < vocab_size:
        merge_counts = {}
        for word in counts.keys():
            count = counts[word]
            tokens = get_tokens(word, vocab)
            for i in range(len(tokens)-1):
                try:
                    merge_counts[(tokens[i], tokens[i+1])] += count
                except:
                    merge_counts[(tokens[i], tokens[i+1])] = count
        (p_m, count) = max(list(merge_counts.items()), key=lambda x: x[1])

        merges.append(p_m)

        vocab_add(vocab, ''.join(p_m))

    return vocab, merges

if __name__ == '__main__':
    word = "based"

    vocab = {'a': {'s': {}}, 'b': {}, 'd': {}, 'e': {}, 's': {}}

    assert get_tokens(word, vocab) == ['b', 'as', 'e', 'd']


    assert vocab_list(vocab) == ['a', 'as', 'b', 'd', 'e', 's']

    vocab_add(vocab, 'attt')

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

    train_bpe(input_path, vocab_size, special_tokens)
    

