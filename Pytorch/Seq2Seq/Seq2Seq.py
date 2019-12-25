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

# %%
lines_path = "Data/movie_lines.txt"
conv_path = "Data/movie_conversations.txt"

# %%
with open(lines_path,'r') as file:
    lines = file.readlines()
for line in lines[:8]:
    print(line.strip())

# %%
