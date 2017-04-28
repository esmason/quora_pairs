"""
Run and evaluate a basic skipthoughts encoder on the quora question pair dataset
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
from sklearn import log_loss

import skipthoughts as st

def load_encoder(model_dir):
    # unfortunately the model_dir param is useless at this point
    model = st.load_model()
    en = st.Encoder(model)
    return en

def load_csv(csv_path):
    # na_filter=False tells pandas to handle empty questions as empty strings, rather than giving them a value of NaN (there are 2 of these in train.csv -- in both cases, question2 is the one that's empty)
    # encoding='utf8' necessary so that nltk tokenizer doesn't throw errors when it encounters exotic characters
    # For more info, see: http://pandas.pydata.org/pandas-docs/stable/io.html#dealing-with-unicode-data
    # ...and on how NLTK handles unicode: http://www.nltk.org/api/nltk.tokenize.html#module-nltk.tokenize
    return pd.read_csv(csv_path, na_filter=False, encoding='utf8')

def load_quora_data(data_dir):
    train_file = os.path.join(data_dir, 'train.csv')
    test_file = os.path.join(data_dir, 'test.csv')
    
    train_table = load_csv(train_file)
    test_table = load_csv(test_file)

    return [train_table, test_table]

def sanitize(sentences):
    # TODO: Implement
    return sentences

def main():
    parser = argparse.ArgumentParser(description='Evaluate a model on the quora question-pair dataset.')
    parser.add_argument('quora_data_path', help='path to the directory containing the quora data')
    parser.add_argument('skipthoughts_model_path', help='path to the directory containing the skipthoughts model')
    args = parser.parse_args()

    print 'Loading quora data...'
    train, test = load_quora_data(args.quora_data_path)

    print 'Loading skipthoughts model...'
    encoder = load_encoder(args.skipthoughts_model_path)

    # # #
    # In the future, fine-tune the model on the training data
    # # #

    print 'Computing train skipthoughts...'
    trainA_vec = encoder.encode(train['question1'], verbose=False, use_eos=True)
    trainB_vec = encoder.encode(train['question2'], verbose=False, use_eos=True)

    print 'Computing test skipthoughts...'
    testA_vec = encoder.encode(test['question1'], verbose=False, use_eos=True)
    testB_vec = encoder.encode(test['question2'], verbose=False, use_eos=True)
    
    # Save the encodings
    print 'Saving the encodings...'
    np.save('trainA_encodings.txt', trainA_vec)
    np.save('trainB_encodings.txt', trainB_vec)
    np.save('testA_encodings.txt', testA_vec)
    np.save('testB_encodings.txt', testB_vec)

    # Evaluate prediction
    labels = test['is_duplicate']
    dists = cosine(testA, testB)

    print "Average distance of questions that are paraphrases:"
    print "     {0}".format(dists[labels == 1])

    print "Averag distance of questions that are NOT paraphrases:"
    print "     {0}".format(dists[labels == 0])    


if __name__ == "__main__":
    main()