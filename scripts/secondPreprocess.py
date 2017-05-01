#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 03:28:17 2017

@author: yanndubois
"""

import numpy as np
import pandas as pd

import enchant 
import difflib

import multiprocessing

def spellCheckSuggest(word):
    if word[0].isupper():
        # if noun just return
        return word
    if d.check(word):
        # if exsits return
        return word
    #if word=="helo":
        # special case that was wrong
     #   return "hello"

    bestWord = word
    bestRatio = 0
    currentRatio = 0
    suggestedWords = set(d.suggest(word))
    for suggestedWord in suggestedWords:
        if ' ' in suggestedWord:
            #don't want to replace with multiple words because wouldn't know what to do with POS
            continue
        currentRatio = difflib.SequenceMatcher(None, word, suggestedWord).ratio()
        if currentRatio > bestRatio:
            bestWord = suggestedWord
            bestRatio = currentRatio
    return bestWord

def preprocessWord(word):
    # returns "number" is it's numeric (I stoped converting it to the real number in strings
    # because it added like 3 4 words everytime) 
    if word.isnumeric():
        return "number"
    word = spellCheckSuggest(word)
    return word


def preprocessPhrase(phrase):
    listWords = [preprocessWord(word) for word in phrase.split()]
    return " ".join(listWords)

def preprocessChunk(chunk):
    chunk['question2'] = chunk['question2'].apply(preprocessPhrase)
    chunk['question1'] = chunk['question1'].apply(preprocessPhrase)
    
    return chunk

def optimizedPreprocess(df):
    # create as many processes as there are CPUs on your machine
    num_processes = multiprocessing.cpu_count() 
    chunks = np.array_split(df, num_processes)
    
    # create our pool with `num_processes` processes
    pool = multiprocessing.Pool(processes=num_processes)
    # apply our function to each chunk in the list
    listResults = pool.map(preprocessChunk, chunks)
    
    # puts everything back together and puts back t oold sorting
    return pd.concat(listResults)

# there were 12 nan!!!
trainK = pd.read_csv("cleanTotalXak.csv").dropna()
trainD = pd.read_csv("cleanXad.csv").dropna()
trainE = pd.read_csv("cleanXae.csv").dropna()
trainF = pd.read_csv("cleanXaf.csv").dropna()
trainG = pd.read_csv("cleanXag.csv").dropna()

allChunks = [trainD,trainE,trainF,trainG]

# which dictionnary
d = enchant.Dict("en_US")

allResults = allChunks
for i in range(0,len(allChunks)):
    chunk = allChunks[i]
    cleanTrain = optimizedPreprocess(chunk)
    cleanTrain.to_csv('cleanChunkBis'+str(i)+".csv", index=False) 
    allResults[i]=cleanTrain

pd.concat(allResults).to_csv('cleanTotalBis.csv', index=False)


