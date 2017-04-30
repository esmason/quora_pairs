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

def sanitize(sentences):
    # TODO: Implement
    return sentences

def encode_skipthoughts(data_file, encoder, out_dir, prefix, verbose):
    print 'Loading quora data...'
    data = du.load_csv(data_file)

    print 'Computing skipthoughts...'
    q1_vecs = encoder.encode(data['question1'], verbose=verbose, use_eos=True)
    q2_vecs = encoder.encode(data['question2'], verbose=verbose, use_eos=True)

    # Save the encodings
    print 'Saving the encodings...'
    np.save(os.path.join(out_dir, prefix + 'q1_encodings'), q1_vecs)
    np.save(os.path.join(out_dir, prefix + 'q2_encodings'), q2_vecs)

def main():
    parser = argparse.ArgumentParser(description='Evaluate a model on the quora question-pair dataset.')
    parser.add_argument('--quora-split-dir', required=True, help='path to the directory containing the quora train, dev and validation splits')
    parser.add_argument('--st-model-dir', required=True, help='path to the directory containing the skipthoughts model')
    parser.add_argument('--output-dir', required=True, help='path to the directory to write to')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()
    verbose = args.verbose

    print 'Loading skipthoughts model...'
    encoder = load_encoder(args.st_model_dir)

    for split_type in ['train', 'dev', 'valid']:
        data_file = os.path.join(args.quora_split_dir, split_type + '.csv')
	encode_skipthoughts(data_file, encoder, args.output_dir, split_type, verbose)


if __name__ == "__main__":
    main()
