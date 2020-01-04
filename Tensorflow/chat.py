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
'''getting the questions and answers  separate'''
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
'''Adding End of string token  to end of every ans'''
for i in range(len(cleaned_answers)):
    cleaned_answers[i]+=' <EOS>'

# %%
'''Translating all  the ques and ans into integer
and replacing  all words  that were filtered oyt bu <OUT>'''
questions_to_int = []
for question in cleaned_questions:
    ints = []
    for word in question.split():
        if word not in questionwords2int:
            ints.append(questionwords2int['<OUT>'])
        else:
            ints.append(questionwords2int[word])
    questions_to_int.append(ints)

answers_to_int = []
for answer in cleaned_answers:
    ints = []
    for word in answer.split():
        if word not in answerwords2int:
            ints.append(answerwords2int['<OUT>'])
        else:
            ints.append(answerwords2int[word])
    answers_to_int.append(ints)  

# %%
answers_to_int[1:5]

# %%
# sorting quest and ans  by length
sorted_clean_questions=[]
sorted_clean_answers=[]
for length in range(1,26):
    for i in enumerate(questions_to_int):
        if(len(i[1])==length):
            sorted_clean_questions.append(questions_to_int[i[0]])
            sorted_clean_answers.append(answers_to_int[i[0]])


# %%
def model_inputs():
    inputs = tf.placeholder(tf.int32,[None,None],name='input')
    targets = tf.placeholder(tf.int32,[None,None],name='target')
    lr = tf.placeholder(tf.float32,name ='learning_rate')
    keep_prob = tf.placeholder(tf.float32,name='keep_prob')
    return inputs,targets,lr,keep_prob

#preprocessing target
def preprocesstargets(targets,word2int,batch_size):
    #batchsize line and 1 column of sos token 
    left_side = tf.fill([batch_size,1],word2int['<SOS>'])
    #contains all answers in batch except last column
    right_side = tf.strided_slice(targets,[0,0],[batch_size,-1],[1,1])
    #making concatination
    preprocessed_targets=tf.concat([left_side,right_side],axis=1)
    return preprocessed_targets

#creating encoder RNN layer
def encoder_rnn(rnn_inputs,rnn_size,num_layers,keep_prob,sequence_length):
    lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
    # lstm = tf.keras.layers.LSTMCell()
    lstm_dropout =  tf.contrib.rnn.DropoutWrapper(lstm,input_keep_prob=keep_prob)
    encoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout]*num_layers)
    encoder_output,encoder_state = tf.nn.bidirectional_dynamic_rnn(cell_fw=encoder_cell,
                                                        cell_bw=encoder_cell,sequence_length=sequence_length,
                                                        inputs=rnn_inputs,dtype=tf.float32)
    return encoder_state

#decoding the training set
def decode_training_set(encoder_state,decoder_cell,decoder_embedded_input,sequence_length,decoding_scope,output_function,keep_prob,batch_size):
    attention_state = tf.zeros([batch_size,1,decoder_cell.output_size])
    attention_keys,attention_values,attention_score_function,attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_state,attention_option='bahdanau',num_units = decoder_cell.output_size)
    #attention_keys- is the key is to be compared with the target states then the attention values
    #attention_values- is the values that we'll use to construct the context vectors.returned by the encoder and that should be used by the decoder as the first element of decoding
    #attention_score_function- used to compute the similarity between the keys and the target
    #attention_construct_function- function used to build the attension state.
    training_decoder_function =  tf.contrib.seq2seq.attention_decoder_fn_train(encoder_state[0],attention_keys,attention_values,attention_score_function,attention_construct_function,name='attention_dec_train')
    decoder_output, decoder_final_state , decoder_final_context_state = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell,training_decoder_function,decoder_embedded_input,sequence_length,decoding_scope)
    decoder_output_dropout = tf.nn.dropout(decoder_output,keep_prob)
    return output_function(decoder_output_dropout)

#decoding for test/validation set

def decode_test_set(encoder_state,decoder_cell,decoder_embeddings_matrix,sos_id,eos_id,max_length,num_words,sequence_length,decoding_scope,output_function,keep_prob,batch_size):
    attention_state = tf.zeros([batch_size,1,decoder_cell.output_size])
    attention_keys,attention_values,attention_score_function,attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_state,attention_option='bahdanau',num_units = decoder_cell.output_size)
    test_decoder_function =  tf.contrib.seq2seq.attention_decoder_fn_inference(output_function,
                                                                               encoder_state[0],
                                                                               attention_keys,
                                                                               attention_values,
                                                                               attention_score_function,
                                                                               attention_construct_function,
                                                                               decoder_embeddings_matrix,
                                                                               sos_id,
                                                                               eos_id,
                                                                               max_length,
                                                                               num_words,
                                                                               name='attention_dec_test')

    test_predictions, decoder_final_state , decoder_final_context_state = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell,
                                                                                                                test_decoder_function,
                                                                                                                decoding_scope)
    return test_predictions

def decoder_rnn(decoder_embedded_input,decoder_embeddings_matrix,encoder_state,num_words,sequence_length,rnn_size, num_layers,word2int,keep_prob,batch_size):
    with tf.variable_scope('decoding') as decoding_scope:
        lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
        lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm,input_keep_prob= keep_prob)
        decoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout]*num_layers)
        weights = tf.truncated_normal_initializer(stddev=0.1)
        biases =tf.zeros_initializer()
        output_function = lambda x: tf.contrib.layers.fully_connected(x,num_words,None,scope = decoding_scope,weights_initializer=weights,biases_initializer=biases)
        training_predictions = decode_training_set(encoder_state,
                                                   decoder_cell,
                                                   decoder_embedded_input,
                                                   sequence_length,
                                                   decoding_scope,
                                                   output_function,
                                                   keep_prob,
                                                   batch_size)
        decoding_scope.reuse_variables()
        test_predictions = decode_test_set(encoder_state,
                                           decoder_cell,
                                           decoder_embeddings_matrix,
                                           word2int['<SOS>'],
                                           word2int['<EOS>'],
                                           sequence_length - 1,
                                           num_words,
                                           decoding_scope,
                                           output_function,
                                           keep_prob,
                                           batch_size)
    return training_predictions, test_predictions























