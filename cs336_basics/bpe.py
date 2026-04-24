def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    TODO:
    """
    pass


# if __name__ == '__main__':
    # import pathlib
    # input_path, vocab_size, special_tokens = ((pathlib.Path(__file__).resolve().parent) / "fixtures",
    #                                           100, '<|endoftext|>')
    
import os
import pathlib
import regex as re

input_path, vocab_size, special_tokens = ((pathlib.Path(os.getcwd()).resolve()) / "tests" / "fixtures" / "corpus.en",
                                              100, '<|endoftext|>')

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

# counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)

# word = "based"

# vocab = {'a': {'s': {}}, 'b': {}, 'd': {}, 'e': {}, 's': {}}

# tokens = []

# for id in range(len(word)):
#     i = id
#     d = vocab
#     while i < len(word):
#         if word[i] in d:
#             d = d[word[i]]
#             i += 1
#         else:
#             break
#     token = word[id:i]
#     tokens.append(token)

def get_tokens(word: str, vocab: dict):

    tokens = []

    for id in range(len(word)):
        i = id
        d = vocab
        while i < len(word):
            if word[i] in d:
                d = d[word[i]]
                i += 1
            else:
                break
        token = word[id:i]
        tokens.append(token)
    
    return tokens

get_tokens(word, vocab)

merge_counts = {}
for word in counts.keys():
    tokens = get_tokens(word, vocab)
    for i in range(len(tokens)-1):
        try:
            merge_counts[(tokens[i], tokens[i+1])] += 1
        except:
            merge_counts[(tokens[i], tokens[i+1])] = 1
merge_counts