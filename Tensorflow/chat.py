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

