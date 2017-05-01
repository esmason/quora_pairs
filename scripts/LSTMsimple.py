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
EMBEDDING_FILE = BASE_DIR + 'name_lol.bin'
Q1_FILE = BASE_DIR + 'q1_lil.bin'
Q2_FILE = BASE_DIR + 'q2_lil.bin'
MAX_SEQUENCE_LENGTH = 30
#cut -d, -f4 dataPos.csv |  awk -v max=0 '{if (NF>max) {max=NF; print NF}}END{print max}'
# max number of words in our data set is 161
MAX_NB_WORDS = 200000
COLUMNS_PREPROCESS = 348
# removes 2 because we have the duplicates and id of question + pairs column too
EMBEDDING_DIM = COLUMNS_PREPROCESS - 3
DEV_SPLIT = 0.1
VALIDATION_SPLIT = 0.1
TRAIN_SPLIT = 1 - DEV_SPLIT - VALIDATION_SPLIT
EMBEDDED_TYPE = np.float32

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

# upload premade embedding
embedded = np.fromfile(EMBEDDING_FILE, dtype=EMBEDDED_TYPE).reshape((COLUMNS_PREPROCESS,-1)).transpose()
# store pair id and duplicate 
pair_id = list(map(int,embedded[:, -1]))
question_id = list(map(int,embedded[:, -2]))
labels = np.array(list(map(int,embedded[:, -3])))

# removes unnecessary column from embedding
embedded = embedded[:, :-3]

data_1 = np.fromfile(Q1_FILE, dtype=EMBEDDED_TYPE).reshape((-1,MAX_SEQUENCE_LENGTH))
data_2 = np.fromfile(Q2_FILE, dtype=EMBEDDED_TYPE).reshape((-1,MAX_SEQUENCE_LENGTH))



########################################
## sample train/validation data
########################################
np.random.seed(1234)
n_pairs = len(data_1)

#perm = np.random.permutation(n_pairs)
#idx_train = perm[:int(n_pairs*(1-VALIDATION_SPLIT-DEV_SPLIT))]
#idx_dev = perm[int(n_pairs*(1-VALIDATION_SPLIT-DEV_SPLIT)):int(n_pairs*(1-VALIDATION_SPLIT))]
#idx_val = perm[int(n_pairs*(1-VALIDATION_SPLIT)):]
idx_train = range(int(n_pairs*(1-VALIDATION_SPLIT-DEV_SPLIT)))
idx_dev = range(int(n_pairs*(1-VALIDATION_SPLIT-DEV_SPLIT)),int(n_pairs*(1-VALIDATION_SPLIT)))
idx_val = range(int(n_pairs*(1-VALIDATION_SPLIT)),n_pairs)

# concatenates them such that the LSTM learns that both LSTM are siamese (same weights)
data_1_train = np.vstack((data_1[idx_train], data_2[idx_train]))
data_2_train = np.vstack((data_2[idx_train], data_1[idx_train]))
labels_train = np.concatenate((labels[idx_train], labels[idx_train]))

data_1_dev = np.vstack((data_1[idx_dev], data_2[idx_dev]))
data_2_dev = np.vstack((data_2[idx_dev], data_1[idx_dev]))
labels_dev = np.concatenate((labels[idx_dev], labels[idx_dev]))

data_1_val = np.vstack((data_1[idx_val], data_2[idx_val]))
data_2_val = np.vstack((data_2[idx_val], data_1[idx_val]))
labels_val = np.concatenate((labels[idx_val], labels[idx_val]))


########################################
## define the model structure
########################################
embedding_layer = Embedding(embedded.shape[0],
        EMBEDDING_DIM,
        weights=[embedded],
        input_length=MAX_SEQUENCE_LENGTH,
        trainable=False)

lstm_layer = LSTM(num_lstm, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm)

sequence_1_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences_1 = embedding_layer(sequence_1_input)
x1 = lstm_layer(embedded_sequences_1)

sequence_2_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences_2 = embedding_layer(sequence_2_input)
y1 = lstm_layer(embedded_sequences_2)

merged = concatenate([x1, y1])
merged = Dropout(rate_drop_dense)(merged)
merged = BatchNormalization()(merged)

merged = Dense(num_dense, activation=act)(merged)
merged = Dropout(rate_drop_dense)(merged)
merged = BatchNormalization()(merged)

preds = Dense(1, activation='sigmoid')(merged)

########################################
## train the model
########################################
model = Model(inputs=[sequence_1_input, sequence_2_input], \
        outputs=preds)
model.compile(loss='binary_crossentropy',
        optimizer='nadam',
        metrics=['acc'])
#model.summary()
print(STAMP)

early_stopping =EarlyStopping(monitor='val_loss', patience=3)
bst_model_path = STAMP + '.h5'
model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True)

hist = model.fit([data_1_train, data_2_train], labels_train, \
        validation_data=([data_1_val, data_2_val], labels_val), \
        epochs=200, batch_size=2048, shuffle=True, \
        callbacks=[early_stopping, model_checkpoint])

model.load_weights(bst_model_path)
bst_val_score = min(hist.history['val_loss'])

########################################
## make the submission
########################################
print('Start making the submission before fine-tuning')

#uses
preds = model.predict([test_data_1, test_data_2], batch_size=8192, verbose=1)
preds += model.predict([test_data_2, test_data_1], batch_size=8192, verbose=1)
preds /= 2

submission = pd.DataFrame({'test_id':test_ids, 'is_duplicate':preds.ravel()})
submission.to_csv('%.4f_'%(bst_val_score)+STAMP+'.csv', index=False)