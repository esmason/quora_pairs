"""
Run and evaluate a basic skipthoughts encoder on the quora question pair dataset
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
from sklearn.metrics import log_loss

import data_utils as du
import kiros.skipthoughts as st

def load_encoder(model_dir):
    model = st.load_model(model_dir)
    en = st.Encoder(model)
    return en

def load_quora_data(data_dir):
    train_file = os.path.join(data_dir, 'train.csv')
    test_file = os.path.join(data_dir, 'test.csv')
    
    train_table = du.load_csv(train_file)
    test_table = du.load_csv(test_file)

    return [train_table, test_table]

def sanitize(sentences):
    # TODO: Implement
    return sentences

def main():
    parser = argparse.ArgumentParser(description='Evaluate a model on the quora question-pair dataset.')
    parser.add_argument('--quora-data-dir', required=True, help='path to the directory containing the quora data')
    parser.add_argument('--st-model-dir', required=True, help='path to the directory containing the skipthoughts model')
    parser.add_argument('--output-dir', default='.', help='path to the directory to write to')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()
    verbose = args.verbose

    print 'Loading quora data...'
    train, test = load_quora_data(args.quora_data_dir)

    print 'Loading skipthoughts model...'
    encoder = load_encoder(args.st_model_dir)

    # # #
    # In the future, fine-tune the model on the training data
    # # #

    print 'Computing train skipthoughts...'
    trainA_vec = encoder.encode(train['question1'], verbose, use_eos=True)
    trainB_vec = encoder.encode(train['question2'], verbose, use_eos=True)

    print 'Computing test skipthoughts...'
    testA_vec = encoder.encode(test['question1'], verbose, use_eos=True)
    testB_vec = encoder.encode(test['question2'], verbose, use_eos=True)
    
    # Save the encodings
    print 'Saving the encodings...'
    np.save('trainA_encodings', trainA_vec)
    np.save('trainB_encodings', trainB_vec)
    np.save('testA_encodings', testA_vec)
    np.save('testB_encodings', testB_vec)

    # Evaluate prediction
    labels = train['is_duplicate']
    dists = np.asarray([cosine(trainA_vec[i], trainB_vec[i]) for i in range(len(trainA_vec))])

    print "Average distance of questions that are paraphrases:"
    print "     {0}".format(np.mean(dists[labels == 1]))

    print "Averag distance of questions that are NOT paraphrases:"
    print "     {0}".format(np.mean(dists[labels == 0]))


if __name__ == "__main__":
    main()
