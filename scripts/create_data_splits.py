import os
import argparse

import data_utils as du

# TODO: Take into account the uneven distribution of labels in the Kaggle public leaderboard, which is approximately 46.3% "is_duplicate" (label of 1)
# More on that can be found here: https://www.kaggle.com/davidthaler/quora-question-pairs/how-many-1-s-are-in-the-public-lb

def main():
    parser = argparse.ArgumentParser(description='Split train.csv into train, dev, and test splits. Specify dev and validation set sizes with args, the remainder is used for training.')
    parser.add_argument('--dataset-file', required=True, help='path to the train.csv file containing the quora training data')
    parser.add_argument('--ndev', type=int, default=1e4, help='size of dev set to create')
    parser.add_argument('--nvalid', type=int, default=5e4, help='size of validation set to create')
    parser.add_argument('--output-dir', required=True, help='directory to which to write train.csv, dev.csv, and valid.csv')
    parser.add_argument('--seed', help='optional random seed to have reproducibility between multiple uses of this tool')
    args = parser.parse_args()

    data = du.load_csv(args.dataset_file)
    shuffled = du.shuffle(data, args.seed)

    ntrain = len(data) - args.ndev - args.nvalid
    train, dev, valid = du.split(shuffled, ntrain, args.ndev, args.nvalid)

    du.write_csv(train, os.path.join(args.output_dir, 'train.csv'))
    du.write_csv(dev, os.path.join(args.output_dir, 'dev.csv'))
    du.write_csv(valid, os.path.join(args.output_dir, 'valid.csv'))


if __name__ == "__main__":
    main()