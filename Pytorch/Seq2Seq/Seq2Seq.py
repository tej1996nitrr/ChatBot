#%%
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as f
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
        print('keep_words {} / {} = {:.4f}'.format(len(keep_words),len(self.word2index,len(keep_words)/len(self.word2index))))
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
pairs
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
        for word in input_sentence.split(''):
            if word not in voc.word2index:
                keep_input = False
                break

        #check output sentence
        for word in output_sentence.split(''):
            if word not in voc.word2index:
                keep_output = False
                break
        #keeping only the pairs  that do not have  trimmed words in their input and output sentences
        if  keep_input and keep_output:
            keep_pairs.append(pair)
    print("Trimmed from {} pairs to {}, {:.4f} of total".format(len(pairs),len(keep_pairs),len(keep_pairs)/len(pairs)))
    return keep_pairs


# %%
