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
'''getting the questios and answers  separate'''
'''in each converation, the first element is the question second will be the answer'''
questions=[]
answers=[]
for conv in conversations_ids:
    for i in range(len(conv)-1):
        questions.append(id2line[conv[i]])
        answers.append(id2line[conv[i+1]])


# %%
def clean_text(text):
    text = text.lower()
    text = re.sub(r"i'm",'i am',text)
    text = re.sub(r"he's",'he is',text)
    text = re.sub(r"she's",'she is',text)
    text = re.sub(r"that's",'that is',text)
    text = re.sub(r"what's",'what is',text)
    text = re.sub(r"where's",'where is',text)
    text = re.sub(r"\'ll",' will',text)
    text = re.sub(r"\'ve",' have',text)
    text = re.sub(r"\'re",' are',text)
    text = re.sub(r"\'d",' would',text)
    text = re.sub(r"won't",' will not',text)
    text = re.sub(r"can't",'cannot',text)
    text = re.sub(r"[-()\"@/#;:<>{}+=~?.,]",'',text)
    return text 

# %%
#cleaning questions
cleaned_questions=[]
for question in questions:
    cleaned_questions.append(clean_text(question))  

#cleaning answers
cleaned_answers=[]
for answer in answers:
    cleaned_answers.append(clean_text(answer)) 


# %%
#creating a dictionary that maps each word to its number of occurences

word2count={}
for question in cleaned_questions:
    for word in question.split():
        if word not in word2count:
            word2count[word] = 1
        else:
           word2count[word] += 1 

for answer in cleaned_answers:
    for word in answer.split():
        if word not in word2count:
            word2count[word] = 1
        else:
           word2count[word] += 1 
        

# %%
'''creating 2 dict that map the question words and the answer words to unique integer'''
'''filtering non frequent words based on threshold ignore words below threshold'''
'''associate words to unique int'''
threshold = 20
questionwords2int = {}
word_number=0
for word,count in word2count.items():
    if count >= threshold:
        questionwords2int[word] = word_number
        word_number+=1  

answerwords2int={}
word_number=0
for word,count in word2count.items():
    if count >= threshold:
        answerwords2int[word] = word_number
        word_number+=1  


# %%
'''adding the last tokens to the two dicts'''
tokens  = ['<PAD>','<EOS>','<OUT>','<SOS>']
for token in tokens:
    questionwords2int[token] = len(questionwords2int)+1

for token in tokens:
    answerwords2int[token] = len(answerwords2int)+1
     

# %%
'''creating the inverse dictionary of the answerwords2int dictionary'''
answersint2word = {w_i : w for  w,w_i in answerwords2int.items()}


# %%
