#%%
from collections import Counter
import json
import torch
import torch.nn as nn
import torch.nn.functional as F 
import math
import torch.utils.data as Dataset 

# %%
lines_path = "Data/movie_lines.txt"
conv_path = "Data/movie_conversations.txt"
max_len = 25
with open(conv_path, 'r') as c:
    conv = c.readlines()
with open(lines_path, 'r') as l:
    lines = l.readlines()

# %%
lines_dic = {}
for line in lines:
    objects = line.split(" +++$+++ ")
    lines_dic[objects[0]] = objects[-1]

# %%
def remove_punc(string):
    punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
    no_punct = ""
    for char in string:
        if char not in punctuations:
            no_punct = no_punct + char  # space is also a character
    return no_punct.lower()

# %%
#making conversations Q/A pairs dictionary
pairs = []
for con in conv:
    ids = eval(con.split(" +++$+++ ")[-1])
    for i in range(len(ids)):
        qa_pairs = []
        
        if i==len(ids)-1:
            break
        
        first = remove_punc(lines_dic[ids[i]].strip())      
        second = remove_punc(lines_dic[ids[i+1]].strip())
        qa_pairs.append(first.split()[:max_len])
        qa_pairs.append(second.split()[:max_len])
        pairs.append(qa_pairs)

# %%
len(pairs)

# %%
word_freq = Counter()
for pair in pairs:
    word_freq.update(pair[0])
    word_freq.update(pair[1])

# %%
min_word_freq = 5
words = [w for w in word_freq.keys() if word_freq[w] > min_word_freq]
word_map = {k: v + 1 for v, k in enumerate(words)}
word_map['<unk>'] = len(word_map) + 1 # for words with frequency less than 5
word_map['<start>'] = len(word_map) + 1
word_map['<end>'] = len(word_map) + 1
word_map['<pad>'] = 0
print("Total words are: {}".format(len(word_map)))

# %%
with open('WORDMAP_corpus.json', 'w') as j:
    json.dump(word_map, j)

# %%
def encode_question(words, word_map):
    enc_c = [word_map.get(word, word_map['<unk>']) for word in words] + [word_map['<pad>']] * (max_len - len(words))
    return enc_c

# %%
encode_question(pairs[0][0],word_map)

# %%
def encode_reply(words, word_map):
    enc_c = [word_map['<start>']] + [word_map.get(word, word_map['<unk>']) for word in words] +     [word_map['<end>']] + [word_map['<pad>']] * (max_len - len(words))
    return enc_c

# %%
pairs_encoded = []
for pair in pairs:
    ques = encode_question(pair[0], word_map)
    ans = encode_reply(pair[1], word_map)
    pairs_encoded.append([ques, ans])
#%%
pairs_encoded[1]
#%%
with open('pairs_encoded.json', 'w') as p:
    json.dump(pairs_encoded, p)
#%%
class Dataset(Dataset):
    def __init__(self):

        self.pairs = json.load(open('pairs_encoded.json'))
        self.dataset_size = len(self.pairs)
        

# %%


# %%


# %%


# %%


# %%


# %%


# %%
