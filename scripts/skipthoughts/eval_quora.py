# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Script to evaluate a skip-thoughts model.

This script can evaluate a model with a unidirectional encoder ("uni-skip" in
the paper); or a model with a bidirectional encoder ("bi-skip"); or the
combination of a model with a unidirectional encoder and a model with a
bidirectional encoder ("combine-skip").

The uni-skip model (if it exists) is specified by the flags
--uni_vocab_file, --uni_embeddings_file, --uni_checkpoint_path.

The bi-skip model (if it exists) is specified by the flags
--bi_vocab_file, --bi_embeddings_path, --bi_checkpoint_path.

The evaluation tasks have different running times. SICK may take 5-10 minutes.
MSRP, TREC and CR may take 20-60 minutes. SUBJ, MPQA and MR may take 2+ hours.
"""

from __future__ import absolute_import
from __future__ import division
# from __future__ import print_function


import tensorflow as tf

from skip_thoughts import configuration
from skip_thoughts import encoder_manager

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("data_dir", None, "Directory containing training data.")
tf.flags.DEFINE_string("output_dir", None, "Directory to which to write logs of performance, and model checkpoints.")

tf.flags.DEFINE_string("uni_vocab_file", None,
                       "Path to vocabulary file containing a list of newline-"
                       "separated words where the word id is the "
                       "corresponding 0-based index in the file.")
tf.flags.DEFINE_string("bi_vocab_file", None,
                       "Path to vocabulary file containing a list of newline-"
                       "separated words where the word id is the "
                       "corresponding 0-based index in the file.")

tf.flags.DEFINE_string("uni_embeddings_file", None,
                       "Path to serialized numpy array of shape "
                       "[vocab_size, embedding_dim].")
tf.flags.DEFINE_string("bi_embeddings_file", None,
                       "Path to serialized numpy array of shape "
                       "[vocab_size, embedding_dim].")

tf.flags.DEFINE_string("uni_checkpoint_path", None,
                       "Checkpoint file or directory containing a checkpoint "
                       "file.")
tf.flags.DEFINE_string("bi_checkpoint_path", None,
                       "Checkpoint file or directory containing a checkpoint "
                       "file.")

tf.logging.set_verbosity(tf.logging.INFO)

'''
Evaluation code for the quora question pairs dataset
'''
import numpy as np
import os
import os.path
from time import time
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import log_loss
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from sklearn.utils import shuffle

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import Adam

import data_utils as du
import logging_utils as log

def evaluate(encoder, output_dir, seed=1234, evaltest=False, loc='./data/'):
    """
    Run experiment

    output_dir specifies distination for saving log file as well as model 

    """
    log_file = os.path.join(output_dir, 'logs.txt')
    with log.FileWriterStdoutPrinter(log_file) as logger:

      print 'Preparing data...'
      train, dev, test, scores = load_data(loc)
      # train[0], train[1], scores[0] = shuffle(train[0], train[1], scores[0], random_state=seed)
      
      print 'Computing training skipthoughts...'
      start = time()
      trainA = encoder.encode(train[0], verbose=False, use_eos=True)
      trainB = encoder.encode(train[1], verbose=False, use_eos=True)
      end = time()
      log.emit_line("Computing skipthoughts for {0} training examples took {1}s".format(len(trainA), end - start)
      
      print 'Computing development skipthoughts...'
      devA = encoder.encode(dev[0], verbose=False, use_eos=True)
      devB = encoder.encode(dev[1], verbose=False, use_eos=True)

      # print 'Computing feature combinations...'
      # TODO: Change so that distance function is a fully-connected neural net where params are shared across indices of the two question vectors
      # trainF = np.c_[np.abs(trainA - trainB), trainA * trainB]
      # devF = np.c_[np.abs(devA - devB), devA * devB]
      trainF = np.c_[trainA, trainB]
      devF = np.c_[devA, devB]

      print 'Encoding labels...'
      trainY = encode_labels(scores[0])
      devY = encode_labels(scores[1])

      print 'Compiling model...'
      lrmodel = prepare_model(ninputs=trainF.shape[1])

      print 'Training...'
      bestlrmodel = train_model(lrmodel, trainF, trainY, 
                                devF, devY, scores[1], 
                                logger, output_dir)

      print 'Saving best model...'
      bestlrmodel.save(os.path.join(output_dir, 'bestlrmodel.h5'))

      if evaltest:
          print 'Computing test skipthoughts...'
          testA = encoder.encode(test[0], verbose=False, use_eos=True)
          testB = encoder.encode(test[1], verbose=False, use_eos=True)

          print 'Computing test feature combinations...'
          # testF = np.c_[np.abs(testA - testB), testA * testB]
          testF = np.c_[testA, testB]

          print 'Evaluating...'
          r = np.arange(0,2)
          yhat = np.dot(bestlrmodel.predict_proba(testF, verbose=2), r)
          pr = pearsonr(yhat, scores[2])[0]
          sr = spearmanr(yhat, scores[2])[0]
          se = mse(yhat, scores[2])
          ll = log_loss(scores[2], np.round(yhat))
          logger.emit_line('Test Pearson: ' + str(pr))
          logger.emit_line('Test Spearman: ' + str(sr))
          logger.emit_line('Test MSE: ' + str(se))
          logger.emit_line('Test log loss: ' + str(ll))

    return yhat


def prepare_model(ninputs=9600, nclass=2):
    """
    Set up and compile the model architecture (Logistic regression)
    """
    lrmodel = Sequential()
    lrmodel.add(Dense(input_dim=ninputs, output_dim=nclass))
    lrmodel.add(Activation('softmax'))
    lrmodel.compile(loss='categorical_crossentropy', optimizer='adam')
    return lrmodel


def encode_labels(labels, nclass=2):
    Y = np.zeros((len(labels), nclass)).astype('int')
    for j, y in enumerate(labels):
      # just use the 0-1 label as an index
      Y[j,y] = 1
    return Y

def train_model(lrmodel, X, Y, devX, devY, devscores, logger, output_dir):
    """
    Train model, using pearsonr on dev for early stopping
    """
    done = False
    best = -1.0
    r = np.arange(0,2)
    chkpt_counter = 0
    
    while not done:
        # Every 100 epochs, check Pearson on development set
        lrmodel.fit(X, Y, verbose=2, shuffle=False, validation_data=(devX, devY))
        yhat = np.dot(lrmodel.predict_proba(devX, verbose=2), r)
        score = pearsonr(yhat, devscores)[0]
        if score > best + 0.01:
            logger.emit_line("({0}) Pearson:{1}".format(chkpt_counter, score))
            ll = log_loss(devY, np.round(yhat)) 
            logger.emit_line("({0}) Log_loss:{1}".format(chkpt_counter, ll))
            chkpt_counter += 1
            if chkpt_counter % 5 == 0:
              lrmodel.save(os.path.join(output_dir, 'lrmodel_{0}.h5'.format(chkpt_counter)))
            best = score
            bestlrmodel = prepare_model(ninputs=X.shape[1])
            bestlrmodel.set_weights(lrmodel.get_weights())
        else:
            done = True

    yhat = np.dot(bestlrmodel.predict_proba(devX, verbose=2), r)
    score = pearsonr(yhat, devscores)[0]
    logger.emit_line('Final Dev Pearson: {0}'.format(score))
    ll = log_loss(devY, np.round(yhat)) 
    logger.emit_line("Final Dev Log_loss: {0}".format(ll))

    return bestlrmodel


def load_data(loc='./data/'):
    """
    Load quora data
    """

    df = du.load_csv(os.path.join(loc, 'train.csv'))
    trainA = df['question1']
    trainB = df['question2']
    trainS = df['is_duplicate']
    
    df = du.load_csv(os.path.join(loc, 'dev.csv'))
    devA = df['question1']
    devB = df['question2']
    devS = df['is_duplicate']

    df = du.load_csv(os.path.join(loc, 'valid.csv'))
    testA = df['question1']
    testB = df['question2']
    testS = df['is_duplicate']

    trainS = [int(s) for s in trainS]
    devS = [int(s) for s in devS]
    testS = [int(s) for s in testS]

    return [trainA, trainB], [devA, devB], [testA, testB], [trainS, devS, testS]

def main(unused_argv):
  if not FLAGS.data_dir:
    raise ValueError("--data_dir is required.")
  if not FLAGS.output_dir:
    raise ValueError("--output_dir is required.")

  encoder = encoder_manager.EncoderManager()

  # Maybe load unidirectional encoder.
  if FLAGS.uni_checkpoint_path:
    print("Loading unidirectional model...")
    uni_config = configuration.model_config()
    encoder.load_model(uni_config, FLAGS.uni_vocab_file,
                       FLAGS.uni_embeddings_file, FLAGS.uni_checkpoint_path)

  # Maybe load bidirectional encoder.
  if FLAGS.bi_checkpoint_path:
    print("Loading bidirectional model...")
    bi_config = configuration.model_config(bidirectional_encoder=True)
    encoder.load_model(bi_config, FLAGS.bi_vocab_file, FLAGS.bi_embeddings_file,
                       FLAGS.bi_checkpoint_path)

  yhat = evaluate(encoder, FLAGS.output_dir, evaltest=True, loc=FLAGS.data_dir)

  encoder.close()


if __name__ == "__main__":
  tf.app.run()
