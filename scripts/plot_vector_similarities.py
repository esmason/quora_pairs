import numpy as np
import pandas as pd
import os

from scipy.spatial.distance import cosine

import seaborn as sns
import matplotlib.pyplot as plt

pal = sns.color_palette()

# vector_dir = '/mnt/d/Education/ubc/courses/term5/cpsc540/project/expt/head1e4'
vector_dir = 'D:\\Education\\ubc\\courses\\term5\\cpsc540\\project\\expt\\head1e4'
print "Reading encodings..."
trainA_vec = np.load(os.path.join(vector_dir, 'trainA_encodings.npy'))
trainB_vec = np.load(os.path.join(vector_dir, 'trainB_encodings.npy'))

# data_dir = '/mnt/d/Education/ubc/courses/term5/cpsc540/project/data/quora/head1e4'
data_dir = 'D:\\Education\\ubc\\courses\\term5\\cpsc540\\project\\data\\quora\\head1e4'

print "Reading csv..."
train = pd.read_csv(os.path.join(data_dir, 'train.csv'), nrows=1e4)
labels = train['is_duplicate']

print "Calculating distances..."
dists = np.asarray([cosine(trainA_vec[i], trainB_vec[i]) for i in range(len(trainA_vec))])

plt.figure()
plt.hist(dists[np.nonzero(labels==1)], bins=50, color=pal[2], label='same')
plt.hist(dists[np.nonzero(labels==0)], bins=50, color=pal[1], alpha=0.5, label='different')
plt.title('Distances for question pairs encoded as skipthought vectors')
plt.legend()
plt.xlabel('Cosine distance', fontsize=15)
plt.ylabel('Number of question pairs', fontsize=15)
plt.show()