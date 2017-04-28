"""
Utility functions for handling the quora dataset using pandas dataframes
"""

import csv
import numpy as np
import pandas as pd

def load_csv(csv_path):
    # na_filter=False tells pandas to handle empty questions as empty strings, rather than giving them a value of NaN (there are 2 of these in train.csv -- in both cases, question2 is the one that's empty)
    # encoding='utf8' necessary so that nltk tokenizer doesn't throw errors when it encounters exotic characters
    # For more info, see: http://pandas.pydata.org/pandas-docs/stable/io.html#dealing-with-unicode-data
    # ...and on how NLTK handles unicode: http://www.nltk.org/api/nltk.tokenize.html#module-nltk.tokenize
    return pd.read_csv(csv_path, na_filter=False, encoding='utf-8')

def write_csv(df, csv_path):
    df.to_csv(csv_path, encoding='utf-8', index=False, quoting=csv.QUOTE_ALL)

def shuffle(df, seed=None):
    """
    Providing a seed gives you the choice of being able to deterministically recreate the same split every time.
    """
    return df.sample(frac=1, random_state=np.random.RandomState(seed)).reset_index(drop=True)

def split(data, train_size, dev_size, valid_size):
    """
    train_size + dev_size + valid_size must not exceed the number of rows in the dataframe.
    """
    if train_size + dev_size + valid_size > len(data):
        raise ValueError('(train_size + dev_size + valid_size) must not exceed dataset size')

    dev_start = train_size
    valid_start = dev_start + dev_size

    # Interestingly, pandas dataframe indexing is inclusive
    train = data.loc[:train_size-1]
    dev = data.loc[train_size:valid_start-1]
    validation = data.loc[valid_start:]

    return (train, dev, validation)
