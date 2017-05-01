#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  1 03:50:42 2017

@author: yanndubois
"""
import numpy as np

BASE_DIR = '../input/'

##### MODIFY HERE #########
idxFirst = 1
idxLast = 6
###########################

EMBEDDING_FILE = BASE_DIR + 'mat_final_1.bin'

Q1_FILE = BASE_DIR + 'q1_final_1.bin'

Q2_FILE = BASE_DIR + 'q2_final_1.bin'

EMBEDDED_TYPE = np.float32
COLUMNS_PREPROCESS = 348
MAX_SEQUENCE_LENGTH = 30
# removes 3 because we have the duplicates and id of question + pairs column too
EMBEDDING_DIM = COLUMNS_PREPROCESS - 3


# adds the embedding matrix
embedded = np.fromfile(EMBEDDING_FILE, dtype=EMBEDDED_TYPE).reshape((COLUMNS_PREPROCESS,-1)).transpose()
for i in range(idxFirst,idxLast):
    embedded = np.concatenate((embedded,
                               np.fromfile(BASE_DIR + 'mat_final_' + str(i) + '.bin', 
                                           dtype=EMBEDDED_TYPE
                            ).reshape((COLUMNS_PREPROCESS,-1)
                            ).transpose()))
    
# adds the embedding matrix
q1 = np.fromfile(Q1_FILE, dtype=np.int32).reshape((-1,MAX_SEQUENCE_LENGTH))
q1b = np.fromfile('../input/q1_final_5.bin', dtype=np.int32).reshape((-1,MAX_SEQUENCE_LENGTH))
for i in range(idxFirst,idxLast):
    q1 = np.stack((q1, 
                   np.fromfile(BASE_DIR + 'mat_final_' + str(i) + '.bin', 
                               dtype=np.int32
                   ).reshape((-1,MAX_SEQUENCE_LENGTH))))
    
# adds the embedding matrix
q2 = np.fromfile(Q2_FILE, dtype=np.int32).reshape((-1,MAX_SEQUENCE_LENGTH))
for i in range(idxFirst,idxLast):
    q2 = np.stack((q2, 
                   np.fromfile(BASE_DIR + 'mat_final_' + str(i) + '.bin', 
                               dtype=np.int32
                   ).reshape((-1,MAX_SEQUENCE_LENGTH))))


embedded.tofile("mat_final.bin")
q1.tofile("q1_final.bin")
q2.tofile("q2_final.bin")


    
    