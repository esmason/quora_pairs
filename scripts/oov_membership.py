"""
Simple script to analyze OOV membership
"""

import os

import cPickle as pkl
from collections import defaultdict

import logging_utils as log

from gensim.models.keyedvectors import KeyedVectors

# needed to unpickle the word dicts we wrote to file
def return_0():
    return 0

PATH_TO_WORD2VEC = '~/models/word2vec/GoogleNews-vectors-negative300.bin'
PATH_TO_FREEBASE = '~/models/word2vec/freebase-vectors-skipgram1000.bin'

PATH_TO_TRAIN_OOV_WORDDICT = '/home/ptcernek/expt/st_oovs2/train_oov_word_freqs.pkl'
PATH_TO_TEST_OOV_WORDDICT = '/home/ptcernek/expt/st_oovs2/test_oov_word_freqs.pkl'

OUTPUT_DIR = '/home/ptcernek/expt/st_oovs2/memberships'
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'membership_stats.txt')

print "Loading word2vec..."
w2v = KeyedVectors.load_word2vec_format(PATH_TO_WORD2VEC, binary=True)
print "Loading freebase..."
freebase = KeyedVectors.load_word2vec_format(PATH_TO_FREEBASE, binary=True)

def retreive_dict(pickle_path):
    pickle_fd = open(pickle_path)
    return pkl.load(pickle_fd)

def analyze_membership(word_dict, writer, prefix):
    still_oov = {}
    in_w2v = {}
    num_in_w2v = 0
    num_in_freebase = 0
    for w, count in word_dict.items():
        found = False
        if w in w2v:
            num_in_w2v += 1
            in_w2v[w] = count
            found = True
        if w in freebase:
            num_in_freebase += 1
            found = True
        if not found:
            still_oov[w] = count
    num_unique_words = len(word_dict)
    num_orig = sum([c for c in word_dict.values()])
    num_still = sum([c for c in still_oov.values()])

    writer.emit_line('Num in w2v={0} ({1}%)'.format(num_in_w2v, num_in_w2v/float(num_unique_words)*100))
    occurrences_in_w2v = sum([c for c in in_w2v.values()])
    writer.emit_line('By frequency these make up {0:.4f}% of the original OOV words'.format(100*float(occurrences_in_w2v)/num_orig))
    writer.emit_line('Num in freebase={0} ({1:.4f}%)'.format(num_in_freebase, num_in_freebase/float(num_unique_words)*100))
    writer.emit_line('Num in neither={0} ({1:.4f}%)'.format(len(still_oov), len(still_oov)/float(num_unique_words)*100))

    writer.emit_line('Of the remaining OOV words, by frequency they make up {0:.4f}% of the original OOV words'.format(num_still/float(num_orig)*100))

    print "Saving words we found in word2vec..."
    with open(os.path.join(OUTPUT_DIR, prefix + '_in_w2v.pkl'), 'w') as fd:
        pkl.dump(in_w2v, fd)

    print "Saving remaining OOV words..."
    with open(os.path.join(OUTPUT_DIR, prefix + '_still_oov.pkl'), 'w') as fd:
        pkl.dump(still_oov, fd)


with log.FileWriterStdoutPrinter(OUTPUT_FILE) as writer:
    print "Retrieving training OOV word dict..."
    train_dict = retreive_dict(PATH_TO_TRAIN_OOV_WORDDICT)
    writer.emit_line("Analyzing OOV membership in training...")
    analyze_membership(train_dict, writer, 'train')

    print "Retrieving test OOV word dict..."
    test_dict = retreive_dict(PATH_TO_TEST_OOV_WORDDICT)
    writer.emit_line("Analyzing OOV membership in testing...")
    analyze_membership(test_dict, writer, 'test')
