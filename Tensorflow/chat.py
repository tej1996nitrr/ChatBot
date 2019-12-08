#%%
import numpy as np 
import re
import time 
import tensorflow as tf

# %%
lines = open('Data/movie_lines.txt',encoding='utf-8',errors='ignore').read().split('\n')
conversations = open('Data/movie_conversations.txt',encoding='utf-8',errors='ignore').read().split('\n')

# %%
'''making dictionary to map each line with id'''

id2line={}
for line in lines:
    line_=line.split(' +++$+++ ')
    if len(line_)==5:
        id2line[line_[0]]=line_[4]
    
    

# %%
'''creating a list of all the conversations'''

conversations_ids = []
for conv in conversations[:-1]:
    conv_ = conv.split(' +++$+++ ')[-1][1:-1].replace("'","").replace(" ","") #take last element
#removed square brackets,'',and spaces
    conversations_ids.append(conv_.split(','))     

# %%
