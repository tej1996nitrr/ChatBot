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
import codecs
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
    #remove less frequent words
    def trim(self,min_count):
        keep_words = []
        for k,v in self.word2count.items():
            if v>=min_count:
                keep_words.append(k)
        print('keep_words {} / {} = {:.4f}'.format(len(keep_words),len(self.word2index,len(keep_words)/len(self.word2index))))




# %%
