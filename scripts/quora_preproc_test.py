"""
This is just a test script.
"""

import os

import pandas as pd
import numpy as np

import nltk
from nltk.tokenize import word_tokenize

quora_data_dir = "/mnt/d/Education/ubc/courses/term5/cpsc540/hw/project_quora/data/quora_data"

quora_train = os.path.join(quora_data_dir, "train.csv")
quora_test = os.path.join(quora_data_dir, "test.csv")

out_dir = "."

def sanitize(text):
    X = []
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    for i, t in enumerate(text, 1):
        try:
            sents = sent_detector.tokenize(t)
        except Exception as e:
            print e
            print "Unable to parse sentence on line {0}: {1}".format(i, t)
            continue
        result = ''
        for s in sents:
            tokens = word_tokenize(s)
            result += ' ' + ' '.join(tokens)
        X.append(result)
    return X

def main():
    print 'Reading data...'
    # na_filter=False tells pandas to handle empty questions as empty strings, rather than giving them a value of NaN (there are 2 of these in train.csv -- in both cases, question2 is the one that's empty)
    # encoding='utf8' necessary so that nltk tokenizer doesn't throw errors when it encounters exotic characters
    # For more info, see: http://pandas.pydata.org/pandas-docs/stable/io.html#dealing-with-unicode-data
    # ...and on how NLTK handles unicode: http://www.nltk.org/api/nltk.tokenize.html#module-nltk.tokenize

    train = pd.read_csv(quora_train, na_filter=False, encoding='utf8')
    # Work only with train to avoid auto-generated questions
    # test = pd.read_csv(quora_test, na_filter=False)

    # Treat entire corpus as one
    # dataset = pd.concat([train, test])
    dataset = train

    print 'Sanitizing question1...'
    cleanedA = sanitize(dataset['question1'])
    print 'Sanitizing question2...'
    cleanedB = sanitize(dataset['question2'])
    for i in range(15):
        print cleanedA[i]
        print cleanedB[i]

    # with open(os.path.join(out_dir, 'cleanedA.txt'), 'w') as fd:
    #     fd.write('\n'.join(cleanedA))
    
    # with open(os.path.join(out_dir, 'cleanedB.txt'), 'w') as fd:
    #     for i, s in enumerate(cleanedB, 1):
    #         try:
    #             fd.write(s + '\n')
    #         except Exception as e:
    #             print e
    #             print "Failed to write sentence on line {0}".format(i, s)
    #             replaced = s.encode('ascii', 'replace')
    #             print "Here's what it looks like replaced: {0}".format(replaced)
    #             fd.write(replaced)

if __name__ == "__main__":
    main()