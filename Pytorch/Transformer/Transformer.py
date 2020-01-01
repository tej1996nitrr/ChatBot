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
# 1, 2, 3, 4, 5, 18240, 18240, 6,
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
class Datasetclass(torch.utils.data.Dataset):

    def __init__(self):

        self.pairs = json.load(open('pairs_encoded.json'))
        self.dataset_size = len(self.pairs)

    def __getitem__(self, i):
        
        question = torch.LongTensor(self.pairs[i][0])
        reply = torch.LongTensor(self.pairs[i][1])
            
        return question, reply

    def __len__(self):
        return self.dataset_size
#%%
# import torch
# class Someclass(torch.utils.data.Dataset):
#     def __init__(self):
#         self.samples = list(range(1, 1001))

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
#         return self.samples[idx]


# %%

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# %%
device

# %%
train_loader = torch.utils.data.DataLoader(Datasetclass(),
                                           batch_size = 100, 
                                           shuffle=True, 
                                           )

# %%
question,reply = next(iter(train_loader))
#iter makes data loader iterable
#next gives a next sample

# %%
print(question.shape)
reply.shape
#we have 2 extra tokens(start and end , so reply is of column lemgth 27)
# %%
'''Creating mask'''
# size=5
# torch.triu(torch.ones(size,size)).transpose(0,1)
#0 is a dimension that will be transformed to dimension 1

def create_masks(question, reply_input, reply_target):
    
    def subsequent_mask(size):
        #we need to create an integer based mask
        mask = torch.triu(torch.ones(size, size)).transpose(0, 1).type(dtype=torch.uint8)
        return mask.unsqueeze(0)
    question_mask = question!=0
    question_mask = question_mask.to(device)
    question_mask = question_mask.unsqueeze(1).unsqueeze(1)         # (batch_size, 1, 1, max_words)
    reply_input_mask = reply_input!=0
    reply_input_mask = reply_input_mask.unsqueeze(1)  # (batch_size, 1, max_words)
    reply_input_mask = reply_input_mask & subsequent_mask(reply_input.size(-1)).type_as(reply_input_mask.data) 

    reply_input_mask = reply_input_mask.unsqueeze(1) # (batch_size, 1, max_words, max_words)
    reply_target_mask = reply_target!=0              # (batch_size, max_words)
    
    return question_mask, reply_input_mask, reply_target_mask



# %%
class Embeddings(nn.Module):
    def __init__(self, vocab_size, d_model, max_len = 50):
        super(Embeddings, self).__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(0.1)
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pe = self.create_positinal_encoding(max_len, self.d_model)
        self.dropout = nn.Dropout(0.1)
        
    def create_positinal_encoding(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model).to(device)
        for pos in range(max_len):   # for each position of the word
            for i in range(0, d_model, 2):   # for each dimension of the each position
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
        pe = pe.unsqueeze(0)   # include the batch size (1,max_len,d_model) 1 will be expanded automatically
        return pe
        
    def forward(self, encoded_words): #overriding the function from module
        embedding = self.embed(encoded_words) * math.sqrt(self.d_model) #(batch_size,max_len,d_model)
        embedding += self.pe[:, :embedding.size(1)]   # pe will automatically be expanded with the same batch size as encoded_words
        embedding = self.dropout(embedding)
        return embedding
# %%

# %%
