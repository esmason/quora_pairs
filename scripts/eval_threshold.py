import os
import argparse

import numpy as np
import sklearn.metrics as sklm

from scipy.spatial.distance import cosine
# import seaborn as sns

import data_utils as du

def find_best_threshold(dists, labels):

    def loss(threshold):
        yhat = dists < threshold
        return sklm.log_loss(labels, yhat)

    candidate_thresholds = np.linspace(dists.min(), dists.max(), num=200)
    vectorized_loss = np.vectorize(loss)
    losses = vectorized_loss(candidate_thresholds)
    best_index = losses.argmin()

    return candidate_thresholds[best_index], losses[best_index]

def plot_histogram(dists, labels, save_dir, threshold=None, log_loss=None, validation=False, show_fig=False):

    import matplotlib

    if not show_fig:
        print "Operating in headless mode, figures will not be shown"
        matplotlib.use('Agg')

    import matplotlib.pyplot as plt

    fig = plt.figure()    

    plt.hist(dists[np.nonzero(labels==1)], bins=50, label='same')
    plt.hist(dists[np.nonzero(labels==0)], bins=50, alpha=0.5, label='different')
    if threshold:
        plt.axvline(threshold)
    
    extra = ''
    if log_loss:
        if validation:
            validation_msg = ' on validation set'
        else:
            validation_msg = ' on training set'
        extra = ' (log_loss={0:.4f}, {1})'.format(log_loss, validation_msg)
    plt.title('Question pair skipthought dists{0}'.format(extra))

    plt.legend()
    plt.xlabel('Cosine similarity', fontsize=15)
    plt.ylabel('Number of question pairs', fontsize=15)
    
    plt.savefig(os.path.join(save_dir, 'hist.png'))

    if not show_fig:
        plt.close(fig)
    else:
        plt.show()


# def try_validation(data_dir, threshold, dists, labels):
#     valid_path = os.path.join(data_dir, 'valid.csv')
#     if not os.path.exists(valid_path):
#         return (False, threshold)

#     valid = du.load_csv(valid_path)
#     return (True, log_loss(labels, dists < threshold))


def write_performance(output_file, threshold, log_loss):
    with open(output_file, 'w') as f:
        f.write('# ' + output_file + '\n')
        f.write('# Cosine similarity' + '\n')
        f.write('threshold=' + str(threshold) + '\n')
        f.write('log_loss=' + str(log_loss) + '\n')


def run_eval(vector_dir, data_dir, output_dir, show_fig=False):

    print "Reading encodings..."
    trainA_vec = np.load(os.path.join(vector_dir, 'trainA_encodings.npy'))
    trainB_vec = np.load(os.path.join(vector_dir, 'trainB_encodings.npy'))

    print "Reading train.csv..."
    train = du.load_csv(os.path.join(data_dir, 'train.csv'))
    labels = train['is_duplicate']

    print "Calculating distances..."
    dists = np.asarray([cosine(trainA_vec[i], trainB_vec[i]) for i in range(len(trainA_vec))])

    print "Finding best threshold distance..."
    threshold, log_loss = find_best_threshold(dists, labels)
    # success, loss = try_validation(data_dir, threshold)
    print "Best threshold={0:.4f}, which induces a log loss of {1:.4f}".format(threshold, log_loss)

    perf_file = os.path.join(output_dir, 'performance.txt')
    print "Saving performance info to " + perf_file
    write_performance(perf_file, threshold, log_loss)

    print "Generating plot and saving it to {0}".format(output_dir)
    plot_histogram(dists, labels, output_dir, threshold=threshold, log_loss=log_loss, show_fig=show_fig)


def main():
    parser = argparse.ArgumentParser(description='Given a bunch of precomputed skipthoughts vectors for pairs of questions, plots a histogram of the cosine similarity between question pairs, finds the threshold cosine similarity that gives the best discrimination between duplicate and non-duplicate questions (measured by log loss), and writes this info to a text file.')
    parser.add_argument('--data-dir', required=True, help='path to the csv file(s) containing the quora training data; if a valid.csv file is found, evaluation will be reported on it')
    parser.add_argument('--vector-dir', required=True, help='path to a directory containing two files called trainA_encodings.npy and trainB_encodings.npy; these should be pickled versions of the skipthoughts vectors computed')
    parser.add_argument('--output-dir', help='directory to which to write histogram and performance info')
    parser.add_argument('--show-fig', action='store_true', help='flag indicating that you want the figure for the histogram plot to be shown. By default it is not shown.')
    args = parser.parse_args()
    args.output_dir = args.output_dir if args.output_dir else args.vector_dir

    run_eval(args.vector_dir, args.data_dir, args.output_dir, args.show_fig)


if __name__ == "__main__":
    main()