#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 29 16:20:04 2017

@author: yanndubois
"""

########################################
## import packages
########################################
import os
import re
import csv
import codecs
import numpy as np
import pandas as pd

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

from gensim.models import KeyedVectors
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers.merge import concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint


########################################
## set directories and parameters
########################################
BASE_DIR = '../input/'
EMBEDDING_FILE = BASE_DIR + 'name.bin'
TRAIN_DATA_FILE = BASE_DIR + 'headCleanPos.csv'
MAX_SEQUENCE_LENGTH = 30
#cut -d, -f4 dataPos.csv |  awk -v max=0 '{if (NF>max) {max=NF; print NF}}END{print max}'
# max number of words in our data set is 161
MAX_NB_WORDS = 200000
COLUMNS_PREPROCESS = 346
# removes 2 because we have the duplicates and id column too
EMBEDDING_DIM = COLUMNS_PREPROCESS - 2
DEV_SPLIT = 0.05
VALIDATION_SPLIT = 0.1
TRAIN_SPLIT = 1 - DEV_SPLIT - VALIDATION_SPLIT
EMBEDDED_TYPE = np.float64

num_lstm = np.random.randint(175, 275)
num_dense = np.random.randint(100, 150)
rate_drop_lstm = 0.15 + np.random.rand() * 0.25
rate_drop_dense = 0.15 + np.random.rand() * 0.25

act = 'relu'
re_weight = False # whether to re-weight classes to fit the 17.5% share in test set
# NOT FOR THE PROJECT BECAUSE NOT USING TEST

STAMP = 'lstm_%d_%d_%.2f_%.2f'%(num_lstm, num_dense, rate_drop_lstm, \
        rate_drop_dense)


########################################
## uploads files
########################################

# uploads preporcessed phrases
data = pd.read_csv(TRAIN_DATA_FILE)[0:12]

# upload premade embedding
embedded = np.fromfile(EMBEDDING_FILE, dtype=np.float64).reshape((COLUMNS_PREPROCESS,-1)).transpose()[0:197]
# store pair id and duplicate 
pair_id_old = list(map(int,embedded[:, -1]))
is_duplicate = embedded[:, -2].tolist()

# removes unnecessary column from embedding
embedded = embedded[:, :-2]

########################################
## padding
########################################
# all sentences have to be the same same length (so use padding)
# and LSTM will learn not to keep the information of words with "only"
# zeros
# note: that we could maybe encode words that have never been seen with 
# only zeros for word2vec but not for the added features => "no meaning"
# but strong features

def unique(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

# multiply by max number of words: new index
pair_id = [item for item in unique(pair_id_old) for i in range(MAX_SEQUENCE_LENGTH)]

def padding (embeddedMatrix):  
    # concatenates the new index
    pandasEmbedded = pd.DataFrame(data=embeddedMatrix).assign(ID=pair_id_old)
    
    embeddedMatrix = np.empty((len(pair_id),embeddedMatrix.shape[1]), dtype=EMBEDDED_TYPE)
    indexList = 0
    index = 0
    modul = 0
    while indexList < embeddedMatrix.shape[0]:
        
        if index == pandasEmbedded.shape[0]:
            # if you come to the end of original but not the list then needs to add 
            # all with padding
            embeddedMatrix[indexList:] = 0
            break
            
        row = pandasEmbedded.loc[index]
        
        # if more than MAX_SEQUENCE_LENGTH delete row
        if pair_id[indexList] > row.ID :
            #increase index in original matrix
            index+=1 
            continue
        
        # if needs padding
        elif pair_id[indexList] < row.ID :
            embeddedMatrix[indexList] = 0
            # if added dowsn't increase index in original matrix
           
        # if needs to keep
        else :
            embeddedMatrix[indexList] = row[:-1]
            #increase index in roiginal matrix
            index+=1 

        # increment index in new matrix only if didn't delete row
        indexList += 1
             
    return embeddedMatrix

embedded=padding(embedded)

#max is 181 and mean is only 16 => lets keep 30 
#Counter(pair_id).most_common(1)[0][1]
#mean(list(Counter(pair_id).values()))


########################################
## sample train/validation/test data
########################################
#np.random.seed(1234)

# finds the ID of last question that will be used as training / dev / validation

index_last_train = int(len(data)*(TRAIN_SPLIT))
index_last_dev = index_last_train + int(len(data)*(DEV_SPLIT))
index_last_validation = len(data)-1

id_last_train = data.loc[index_last_train].id
id_last_dev = data.loc[index_last_dev].id
id_last_validation = data.loc[index_last_validation].id

# finds the indices of embedded vector that will be used as training / dev / validation

index_last_train_embedded = len(pair_id) - 1 - pair_id[::-1].index(id_last_train)
index_last_dev_embedded = len(pair_id) - 1 - pair_id[::-1].index(id_last_dev)
index_last_validation_embedded = len(pair_id) - 1 - pair_id[::-1].index(id_last_validation)

# simple numpy matrix that makes each word point to its correseponding word
# we already computed matrix => simply a range with 30 columns per rows

embedded_train = np.arange(index_last_train_embedded+1).reshape((-1,MAX_SEQUENCE_LENGTH))
embedded_dev = np.arange(index_last_train_embedded+1,index_last_dev_embedded+1).reshape((-1,MAX_SEQUENCE_LENGTH))
embedded_validation = np.arange(index_last_dev_embedded+1,index_last_validation_embedded+1).reshape((-1,MAX_SEQUENCE_LENGTH))

# separates between questions => odd and even
data_1_train = embedded_train[::2]
data_1_dev = embedded_dev[::2]
data_1_val = embedded_validation[::2]

data_2_train = embedded_train[1::2]
data_2_dev = embedded_dev[1::2]
data_2_val = embedded_validation[1::2]

del embedded_train
del embedded_dev
del embedded_validation

labels_train = data.loc[0:index_last_train].is_duplicate
labels_dev = data.loc[index_last_train+1:index_last_dev].is_duplicate
labels_val= data.loc[index_last_dev+1:index_last_validation].is_duplicate

#del embedded

########################################
## define the model structure
########################################
embedding_layer = Embedding(embedded.shape[0],
        EMBEDDING_DIM,
        weights=[embedded],
        input_length=MAX_SEQUENCE_LENGTH,
        trainable=False)

lstm_layer = LSTM(num_lstm, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm)

sequence_1_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int64')
embedded_sequences_1 = embedding_layer(sequence_1_input)
x1 = lstm_layer(embedded_sequences_1)

sequence_2_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int64')
embedded_sequences_2 = embedding_layer(sequence_2_input)
y1 = lstm_layer(embedded_sequences_2)

merged = concatenate([x1, y1])
merged = Dropout(rate_drop_dense)(merged)
merged = BatchNormalization()(merged)

merged = Dense(num_dense, activation=act)(merged)
merged = Dropout(rate_drop_dense)(merged)
merged = BatchNormalization()(merged)

preds = Dense(1, activation='sigmoid')(merged)

weight_val = np.ones(len(labels_val))
class_weight = None

########################################
## train the model
########################################
model = Model(inputs=[sequence_1_input,  v], \
        outputs=preds)
model.compile(loss='binary_crossentropy',
        optimizer='nadam',
        metrics=['acc'])
#model.summary()
print(STAMP)

early_stopping =EarlyStopping(monitor='val_loss', patience=3)
bst_model_path = STAMP + '.h5'
model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True)

hist = model.fit([data_1_train, data_2_train], labels_train[0:6], \
        validation_data=([data_1_val, np.stack([data_2_val,data_1_val[0]])], labels_val, weight_val), \
        epochs=200, batch_size=2048, shuffle=True, \
        class_weight=class_weight, callbacks=[early_stopping, model_checkpoint])

model.load_weights(bst_model_path)
bst_val_score = min(hist.history['val_loss'])