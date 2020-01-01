#%%
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import csv
import random
import re
import os
import itertools
import unicodedata
import codecs

# %%
CUDA = torch.cuda.is_available()
device = torch.device("cuda" if CUDA else "cpu")

# %%
device
#%%
'''Data Preprocessing'''

lines_path = "Data/movie_lines.txt"
conv_path = "Data/movie_conversations.txt"

with open(lines_path,'r') as file:
    lines = file.readlines()
for line in lines[:8]:
    print(line.strip())


# %%
# split each line of the file into  a dictionary of fields(lineID,charachterID,movieID,charachter,text)
line_fields = ["lineID","characterID","movieID","character","text"]
lines = {}
with open(lines_path,'r',encoding='iso-8859-1') as f:
    for line in f:
        values = line.split(" +++$+++ ")
        #extracting fields
        lineObj = {}
        for i, field in enumerate(line_fields):
            lineObj[field] = values[i]
        lines[lineObj['lineID']] = lineObj


# %%
'''
lines['L872'] = {'lineID': 'L872',
 'characterID': 'u0',
 'movieID': 'm0',
 'character': 'BIANCA',
 'text': "Okay -- you're gonna need to learn how to lie.\n"}
 '''

# %%

#groups fields of lines from 'Loadlines' into conversations based on"movie_conversations.txt"
conv_fields = ["characterID","character2ID","movieID","utteranceIDs"]
conversations = []
with open(conv_path,'r',encoding='iso-8859-1') as f:
    for l in f:
        values = l.split(" +++$+++ ")
        convObj = {}
        for i ,field in enumerate(conv_fields):
            convObj[field] = values[i]
        #converting string result from split to list
        linesIDs = eval(convObj['utteranceIDs'])
        # print(convObj['utteranceIDs'])
        #reassembling lines
        convObj['lines']=[]
        for lineID in linesIDs:
            convObj['lines'].append(lines[lineID])
        conversations.append(convObj)



# %%

'''
conversations[0]=

{'characterID': 'u0',
 'character2ID': 'u2',
 'movieID': 'm0',
 'utteranceIDs': "['L194', 'L195', 'L196', 'L197']\n",
 'lines': [{'lineID': 'L194',
   'characterID': 'u0',
   'movieID': 'm0',
   'character': 'BIANCA',
   'text': 'Can we make this quick?  Roxanne Korrine and Andrew Barrett are having an incredibly horrendous public break- up on the quad.  Again.\n'},
  {'lineID': 'L195',
   'characterID': 'u2',
   'movieID': 'm0',
   'character': 'CAMERON',
   'text': "Well, I thought we'd start with pronunciation, if that's okay with you.\n"},
  {'lineID': 'L196',
   'characterID': 'u0',
   'movieID': 'm0',
   'character': 'BIANCA',
   'text': 'Not the hacking and gagging and spitting part.  Please.\n'},
  {'lineID': 'L197',
   'characterID': 'u2',
   'movieID': 'm0',
   'character': 'CAMERON',
   'text': "Okay... then how 'bout we try out some French cuisine.  Saturday?  Night?\n"}]}
   '''

# %%
'''Arranging data in qna format'''
qa_pairs=[]
for conversation in conversations:
    for i in range(len(conversation['lines'])-1):
        inputLine = conversation['lines'][i]['text'].strip()
        targetLine = conversation['lines'][i+1]['text'].strip()
        if inputLine and targetLine:
            qa_pairs.append([inputLine,targetLine])




#%%
'''Saving processed text'''

#%%
delimeter='\t'
datafile = 'formatted_movie_lines.txt'
delimeter = str(codecs.decode(delimeter,"unicode_escape"))
with open(datafile,'w',encoding='utf-8') as outputfile:
    writer= csv.writer(outputfile,delimiter=delimeter)
    for pairs in qa_pairs:
        writer.writerow(pairs)

# print("Done writing")
# with open(datafile,'rb') as file:
#     lines = file.readlines()
# for line in lines[:8]:
#     print(line)


# %%

'''Processing words'''

PAD_token = 0
SOS_token = 1
EOS_token = 2
class vocabulary:
    def __init__(self,name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token:"PAD",SOS_token:"SOS",EOS_token:"EOS"}
        self.num_words = 3
    
    def addSentence(self,sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self,word):
        if  word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words+=1
        else:
            self.word2count[word] +=1

    #remove less frequent words below threshold
    def trim(self,min_count):
        keep_words = []
        for k,v in self.word2count.items():
            if v>=min_count:
                keep_words.append(k)
        print('keep_words {} / {} = {:.4f}'.format(len(keep_words),len(self.word2index),len(keep_words)/len(self.word2index)))
        self.word2count = {}
        self.index2word = {PAD_token:"PAD",SOS_token:"SOS",EOS_token:"EOS"}
        self.word2index = {}
        self.num_words = 3
        for word in keep_words:
            self.addWord(word)

# %%
def unicodeToAscii(s):
    return ''.join(c for c in unicodedata.normalize('NFD',s) if unicodedata.category(c)!='Mn')

# %%
# unicodeToAscii('Montréal')

# %%
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])",r" \1",s) #
    s = re.sub(r"[^a-zA-Z.!?]+",r" ",s)# remove any other character other tha lower and upper case letters
    s = re.sub(r"\s+",r" ",s).strip()
    return s

# %%
# normalizeString("aa123!s's dd?")

# %%
lines = open(datafile,encoding='utf-8').read().strip().split('\n')
pairs = [[normalizeString(s) for s in pair.split('\t')] for pair in lines]
print("Done Reading")
#%%
voc = vocabulary('Movie')

# %%
#returns true if both sentences in a pair 'p' are under MAX_LENGTH threshold
MAX_LENGTH = 10 
#maximum  sentence length to consider(max words)
def filterPair(p):
    #input sequence need to preserve  the last word for EOS token
    return len(p[0].split()) < MAX_LENGTH and len(p[1].split())< MAX_LENGTH

def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

# %%
# len(pairs)
# pairs
pairs_backup = pairs
# %%
pairs = [pair for pair in pairs if len(pair)>1]
pairs = filterPairs(pairs)

# %%
len(pairs)

# %%
#looping through each pair and add  question and reply to the vocabulary
for pair in pairs:
    voc.addSentence(pair[0])
    voc.addSentence(pair[1])
print("Counted Words ",voc.num_words)
for pair in pairs[:10]:
    print(pair)


# %%

'''Trimming rare words'''
MIN_COUNT = 3
def trimRareWords(voc,pairs,MIN_COUNT):
    
    voc.trim(MIN_COUNT)
    keep_pairs = []
    for pair in pairs:
        input_sentence = pair[0]
        output_sentence = pair[1]
        keep_input = True
        keep_output = True

        #check input sentence
        for word in input_sentence.split(' '):
            if word not in voc.word2index:
                keep_input = False
                break

        #check output sentence
        for word in output_sentence.split(' '):
            if word not in voc.word2index:
                keep_output = False
                break
        #keeping only the pairs  that do not have  trimmed words in their input and output sentences
        if  keep_input and keep_output:
            keep_pairs.append(pair)
    # print("Trimmed from {} pairs to {}, {:.4f} of total".format(len(pairs),len(keep_pairs),len(keep_pairs)/len(pairs)))
    return keep_pairs


# %%
pairs = trimRareWords(voc,pairs,MIN_COUNT)

# %%
# pair

# %%
'''Data Preparation'''
'''Converting words to index'''
def indexesFromSentences(voc,sentence):
    return [voc.word2index[word] for word in sentence.split(' ')]+[EOS_token]


# %%
pairs[1][0]

# %%
# indexesFromSentences(voc,pairs[1][0])

# %%
inp = []
out = []
i = 0
for pair in pairs[:10]:
    inp.append(pair[0])
    out.append(pair[1])
print(inp)
print(len(inp))
indexes = [indexesFromSentences(voc,s) for s in inp]
indexes

# %%
def zeropadding(l,fillvalue=0):
    return list(itertools.zip_longest(*l,fillvalue=fillvalue))

# %%
leng = [len(ind) for ind in indexes]
max(leng)
#%%
# import itertools
# a=[[3, 4, 2],
#  [7, 8, 9, 10, 4, 11, 12, 13, 2],
#  [16, 4, 2],
#  [8, 31, 22, 6, 2],
#  [33, 34, 4, 4, 4, 2],
#  [35, 36, 37, 38, 7, 39, 40, 41, 4, 2],
#  [42, 2],
#  [47, 7, 48, 40, 45, 49, 6, 2],
#  [50, 51, 52, 6, 2],
#  [58, 2]]

# list(itertools.zip_longest(*a,fillvalue=0))

# %%
test_result = zeropadding(indexes)
print(len(test_result))
test_result
#rows = max_length
#columns = batch_size (for testing =10)
# %%
def binaryMatrix(l,value=0):
    m = []
    for i,seq in enumerate(l):
        m.append([])
        for token in seq:
            if token == PAD_token:
                m[i].append(0)
            else:
                m[i].append(1)
    return m


# %%
binary_result = binaryMatrix(test_result)
binary_result
# %%

#returns padded input sequence tensor 
def inputvar(l,voc):
    indexes_batch = [indexesFromSentences(voc,sentence) for sentence in l]
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    padList = zeropadding(indexes_batch)
    padVar = torch.LongTensor(padList)
    return padVar,lengths

#returns padded target sequence tensor, padding mask, and mask target length
def outputvar(l,voc):
    indexes_batch = [indexesFromSentences(voc,sentence) for sentence in l]
    max_target_len = max([len(indexes) for indexes in indexes_batch])
    padList = zeropadding(indexes_batch)
    mask = binaryMatrix(padList)
    mask = torch.ByteTensor(mask)
    padVar = torch.LongTensor(padList)
    return padVar, mask, max_target_len

#returns all items for given batch of pairs
def batch2TrainData(voc, pair_batch):
    #sort questions in descending order length
    pair_batch.sort(key = lambda x:len(x[0].split(" ")), reverse = True)
    input_batch,output_batch=[],[]
    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch.append(pair[1])
    inp, length  = inputvar(input_batch,voc)
    output, mask, max_target_len = outputvar(output_batch,voc)
    return inp, length, output, mask, max_target_len

import random
small_batch_size = 5
batches = batch2TrainData(voc,[random.choice(pairs) for _ in range(small_batch_size)])
input_variable,lengths, target_variables, mask, mask_target_len = batches
print("input_variables")
print(input_variable)
print("lengths",lengths)
print("target_variables",target_variables)
print("mask",mask)
print("mask_target_len",mask_target_len)


# %%
'''Defining ENcoder'''
class EncoderRNN(nn.Module):
    def __init__(self, hidden_size, embedding, n_layers=1, droput=0):
        super(EncoderRNN,self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = embedding
        #initializing gru. input_size and hidden_size are set to hidden_size
        #because our input size is a word embedding with number_of_features == hidden_size
        #hidden_size = no. of neurons in hidden layer = no. of rnn cells
        self.gru = nn.GRU(hidden_size,hidden_size,n_layers, dropout = (0 if n_layers==1 else droput),bidirectional=True) 

    def forward(self,input_seq,input_lengths, hidden=None):
        #input_seq = batch of input sequences; shape = (max_length,batch_size)
        #inputs_length = list of sentence lengths corresponding to each sentence in the batch
        #hidden_state,of shape:(n_layers x num_directions,batch_size,hidden_size)
        #convert word indexes to embeddings
        embedded = self.embedding(input_seq)
        #pack padded batch of sequence for RNN module
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded,input_lengths)
        #forward pass through GRu
        outputs, hidden = self.gru(packed,hidden)
        #unpack padding
        outputs, _= torch.nn.utils.rnn.pad_packed_sequence(outputs)
        #sum bidirectional GRU outputs
        outputs = outputs[:,:,:self.hidden_size]+outputs[:,:,self.hidden_size:]
        return outputs,hidden
        #outputs:the output features h_t from the last layer of the gru, for each timestep(sum of bidirectional outputs)
        #outputs shape = (max_length, batch_size, hidden_size)]
        #hidden: hidden state for the last timestep of shape( n_layers x num_directions, batch_size, hidden_size)


#%%
# a = torch.randn(6,7) #6 batches of max 7 words
# lengths = [7,7,6,5,4,2] #length of each batch
# targets = torch.nn.utils.rnn.pack_padded_sequence(a,lengths,batch_first=True)
# print(targets[0].shape)
# print(a)
# print(targets[0])
# print(targets[1])

'''
a = torch.randn(6,7) #6 batches of max 7 words...
torch.Size([31])
tensor([[ 0.0894,  0.0223,  0.7523,  0.8689,  0.8928,  0.7619,  1.1048],
        [-1.0012,  1.8771, -1.4949, -0.6938, -0.8964, -1.1081,  0.5691],
        [-0.7850, -0.2666, -1.0998,  1.2676,  2.3676, -0.6189,  0.3998],
        [-0.4090, -1.7571, -0.5955,  0.8331,  1.4483, -0.3819, -1.1568],
        [ 0.5162, -0.4082, -1.3798, -0.6103, -0.3605,  1.0129, -0.4235],
        [ 0.9563, -0.0239,  1.1315,  1.8441, -1.7678,  0.5480, -1.2487]])
tensor([ 0.0894, -1.0012, -0.7850, -0.4090,  0.5162,  0.9563,  0.0223,  1.8771,
        -0.2666, -1.7571, -0.4082, -0.0239,  0.7523, -1.4949, -1.0998, -0.5955,
        -1.3798,  0.8689, -0.6938,  1.2676,  0.8331, -0.6103,  0.8928, -0.8964,
         2.3676,  1.4483,  0.7619, -1.1081, -0.6189,  1.1048,  0.5691])
tensor([6, 6, 5, 5, 4, 3, 2])'''


# %%
class Attn(torch.nn.Module):
    def __init__(self, method, hidden_state):
        super(Attn,self).__init__()
        self.method = method
        self.hidden_size = hidden_size

    def dot_score(self, hidden, encoder_output):
        return torch.sum(hidden * encoder_output, dim=2) # second dim = hidden_size
    
    def forward(self,hidden, encoder_outputs):
        #hidden of shape:(1, batch_size, hidden_size)
        #encoder_outputs of shape: (max_length,batch_size,hidden_size)
        # (1, batch_size, hidden_size) * (max_length,batch_size,hidden_size) = (max_length,batch_size,hidden_size)
        attn_energies = self.dot_score(hidden,encoder_outputs)        #(max_length,batch_size)
        attn_energies = attn_energies.t()                             #(batch_size,max_length)
        return F.softmax(attn_energies,dim=1).unsqueeze(1)            #(batch_size,1,max_length)


