#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 03:28:17 2017

@author: yanndubois
"""

import numpy as np
import pandas as pd
import multiprocessing

#from num2words import num2words
from unidecode import unidecode

import nltk
from nltk.corpus import stopwords

from nltk.tag import StanfordNERTagger
from nltk.tag import StanfordPOSTagger
import re






##### PREPROCESSING #####
# which dictionnary


# the NER and POS tagger
nertagger = StanfordNERTagger("../downloads/stanford-ner-2016-10-31/classifiers/english.all.3class.distsim.crf.ser.gz",
                                      "../downloads/stanford-ner-2016-10-31/stanford-ner.jar")
postagger = StanfordPOSTagger("../downloads/stanford-postagger-2016-10-31/models/english-bidirectional-distsim.tagger",
                                      "../downloads/stanford-postagger-2016-10-31/stanford-postagger.jar")

def replaceKM(match):
        if match.group(2) == "K" or match.group(2) == "k":
            return str(int(match.group(1))) + "000"
        return str(int(match.group(1))) + "000000"
    
def replaceEXP(match):
    return str(int(match.group(1))) + str(10 ** int(match.group(2)))[1:]
    
def cleaningPOS(phrase, isReplaceNumber=False):
    
    # Replace special words: ideas taken from
    # https://www.kaggle.com/currie32/quora-question-pairs/the-importance-of-cleaning-text/notebook
    
    phrase = re.sub(r" [U|u]sa ", " America ", phrase)
    phrase = re.sub(r" USA ", " America ", phrase)
    phrase = re.sub(r" [U|u]ssr ", " Russia ", phrase)
    phrase = re.sub(r" USSR ", " Russia ", phrase)
    phrase = re.sub(r" the US ", " America ", phrase)
    phrase = re.sub(r" [U|u]k ", " England ", phrase)
    phrase = re.sub(r" UK ", " England ", phrase)
    phrase = re.sub(r" [U|u]nited [K|k]ingdom ", " England ", phrase)
    phrase = re.sub(r" [G|g]reat [B|b]ritain ", " England ", phrase)
    phrase = re.sub(r" india ", " India ", phrase)
    phrase = re.sub(r" switzerland ", " Switzerland ", phrase)
    phrase = re.sub(r" england ", " England ", phrase)
    phrase = re.sub(r" brazil ", " Brazil ", phrase)
    phrase = re.sub(r" canada ", " Canada ", phrase)
    phrase = re.sub(r" germany ", " Germany ", phrase)
    phrase = re.sub(r" italy ", " Italy ", phrase)
    phrase = re.sub(r" russia ", " Russia ", phrase)
    phrase = re.sub(r" china ", " China ", phrase)
    
    phrase = re.sub(r"([0-9| ])[K|k]ms ", r"\1 kilometers ", phrase)
    phrase = re.sub(r"([0-9| ])KMs ", r"\1 kilometers ", phrase)
    phrase = re.sub(r"([0-9| ])KM ", r"\1 kilometers ", phrase)
    phrase = re.sub(r"([0-9| ])[K|k]ms ", r"\1 kilometers ", phrase)
    phrase = re.sub(r"([0-9| ])[K|k]m ", r"\1 kilometers ", phrase)
    phrase = re.sub(r"([0-9| ])[C|c] ", r"\1 celsius ", phrase)
    phrase = re.sub(r"([0-9| ])[F|f] ", r"\1 celsius ", phrase)
    phrase = re.sub(r"([0-9])m ", r"\1 meters ", phrase)
    phrase = re.sub(r"([0-9])g ", r"\1 Gram ", phrase)
    phrase = re.sub(r"([0-9| ])[K|k]g ", r"\1 kilogram ", phrase)
    phrase = re.sub(r"([0-9| ])[L|l]b ", r"\1 pound ", phrase)
    phrase = re.sub(r"([0-9])s ", r"\1 seconds ", phrase)
    phrase = re.sub(r"([0-9| ])[M|m]in ", r"\1 minutes ", phrase)
    phrase = re.sub(r"([0-]9)[H|h] ", r"\1 hours ", phrase)
    phrase = re.sub(r"([0-9])[c|C]m ", r"\1 centimeters ", phrase)
    phrase = re.sub(r"([0-9])mm ", r"\1 millimiters ", phrase)
    phrase = re.sub(r"([0-9])ft ", r"\1 feet ", phrase)
    phrase = re.sub(r"([0-9])L ", r"\1 liters ", phrase)
    phrase = re.sub(r"([0-9| ])[K|k][B|b] ", r"\1 kilobyte ", phrase)
    phrase = re.sub(r"([0-9| ])[M|m][B|b] ", r"\1 megabyte ", phrase)
    phrase = re.sub(r"([0-9| ])[G|g][B|b] ", r"\1 gigabyte ", phrase)
    phrase = re.sub(r"([0-9])[T|t][B|b] ", r"\1 terabyte ", phrase)
    phrase = re.sub(r"([0-9])([K|k|M]) ", replaceKM, phrase)
    phrase = re.sub(r"([0-9])e([0-9]+) ", replaceEXP, phrase) 
    
    phrase = re.sub(r" quora ", " Quora ", phrase)
    phrase = re.sub(r" dms ", "direct messages ", phrase)  
    phrase = re.sub(r" Apple ", " Microsoft ", phrase) # close enough and better than the fruit
    phrase = re.sub(r" cs ", " computer science ", phrase) 
    phrase = re.sub(r" cpcs ", " computer science ", phrase) 
    phrase = re.sub(r" upvotes ", " up votes ", phrase)
    phrase = re.sub(r" [i|I][P|p]hone ", " phone ", phrase)
    phrase = re.sub(r" ios ", " operating system ", phrase)
    phrase = re.sub(r" [G|g]ps ", " GPS ", phrase)
    phrase = re.sub(r" gst ", " GST ", phrase)
    phrase = re.sub(r" i[P|p]ad ", " tablet ", phrase)
    phrase = re.sub(r" i[P|p]hone ", " phone ", phrase)
    
    phrase = re.sub(r" bestfriend ", " best friend ", phrase)
    phrase = re.sub(r" dna ", " DNA ", phrase)
    phrase = re.sub(r"III", "3", phrase)
    phrase = re.sub(r"II", "2", phrase) 

    
    # Replace apostrophe that cut words
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    
    # maps special letters to there unicode counterpart
    phrase = unidecode(phrase)
    
    # Removes special characters
    phrase = re.sub(r"[^a-zA-Z0-9| ]", "", phrase) 
    
    # Removes to many spaces
    phrase = re.sub("  ", " ", phrase)
    
    # Numbers to words and remove first 
    arrayPhrase = phrase.split()
    phrase=""
    
    if isReplaceNumber:
        #probably better not using because adds many words!!
        for index, word in enumerate(arrayPhrase):
            if word.isnumeric():
                # often gives back hyphen => deletes
                # chooses to make int but could keep float
                word = ""#num2words(int(word))
            phrase = phrase + " " + word
    
        # Removes to many spaces
        phrase = re.sub("  ", " ", phrase)
        arrayPhrase = phrase.split()
    

    
    # Computes NER (POS for nouns): can give Person / Organization / Location
    pos = [x[1] for x in postagger.tag(arrayPhrase)]
    
    # Computes NER (POS for nouns): can give Person / Organization / Location
    ner = [x[1] for x in nertagger.tag(arrayPhrase)]
    
    # Special pos that take into account the ner
    pos = [pos[i] if x=="O" else x for i,x in enumerate(ner)]
    
    # Removes stop words
    for index, word in enumerate(arrayPhrase):
        if word in stopwords.words('english'):
            del arrayPhrase[index] 
            del pos[index]   
            
    # now that tagged can put all to lower case
    return " ".join(arrayPhrase).lower(), " ".join(pos)


def preprocessChunk(chunk):
    # Add the string 'empty' to empty strings
    chunk = chunk.fillna('empty')
    #prepare columns
    chunk["POS1"] = ""
    chunk["POS2"] = ""
    indices = chunk.index
    # use cython loop
    phrase1 = ""
    phrase2= ""
    oldPhrase1= ""
    oldPhrase2= ""
    oldPOS1= ""
    oldPOS2= ""
    oldPhrase1final= ""
    oldPhrase2final= ""
    
    for i in range(chunk.shape[0]):
        phrase1 = chunk.loc[indices[i],'question1']
        
        # only computes if different from previous
        if phrase1 != oldPhrase1 :
            oldPhrase1final, oldPOS1 = cleaningPOS(phrase1)
        
        chunk.set_value(indices[i], 'question1', oldPhrase1final)
        chunk.set_value(indices[i], 'POS1', oldPOS1)
        oldPhrase1 = phrase1
        
        phrase2 = chunk.loc[indices[i],'question2']
        
        # only computes if different from previous
        if phrase2 != oldPhrase2 :
            oldPhrase2final, oldPOS2 = cleaningPOS(phrase2)
        
        chunk.set_value(indices[i], 'question2', oldPhrase2final)
        chunk.set_value(indices[i], 'POS2', oldPOS2)
        oldPhrase2 = phrase2
   
    # return the chunk!
    return chunk


def preprocessPOS(df):
    # sort to make quicker for duplicates
    if sum(train.duplicated(['question1'])) > sum(train.duplicated(['question2'])):
        # sorts by the column where the most duplicates
        df = df.sort_values('question1')
    else :
        df = df.sort_values('question2')
    
    # create as many processes as there are CPUs on your machine
    num_processes = multiprocessing.cpu_count() 
    chunks = np.array_split(df, num_processes)
    
    # create our pool with `num_processes` processes
    pool = multiprocessing.Pool(processes=num_processes)
    # apply our function to each chunk in the list
    listResults = pool.map(preprocessChunk, chunks)
    
    # puts everything back together and puts back t oold sorting
    return pd.concat(listResults).sort_index()
    


train = pd.read_csv("../data/train.csv")
numberChunks = 10
allChunks = np.array_split(train, numberChunks)
allResults = allChunks
for i in range(0,numberChunks):
    chunk = allChunks[i]
    cleanTrain = preprocessPOS(chunk)
    cleanTrain.to_csv('../data/cleanChunk'+str(i), index=False) 
    allResults[i]=cleanTrain
pd.concat(allResults).to_csv('../data/cleanTotal.csv', index=False)
