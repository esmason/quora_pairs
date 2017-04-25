import tensorflow as tf
from word2vec_utils import one_question_distances, two_question_distances, row_and_col_mins, get_POS_dict, pos_to_list
import csv
import numpy as np
from csv import DictReader
from word2vec_loader import load_word_vectors

filename = "../data/cleanTrainHead.csv"

#dist_mat_test = np.array([[0,1],[2,3]])
#print(row_and_col_mins(dist_mat_test))
word_vectors = load_word_vectors()
POS_dict = get_POS_dict()

def assemble_row_inputs(row):
	'''returns a matrix where each column is a word in order [q1 words, q2 words]
		each row is a feature from top to bottom POS/NER tag, inter min cos distance,
		inter min city-block distance, intra min cos distance (nonzero), intra min
		city block distance (nonzero only)'''

	q1 = row['question1']
	q2 = row['question2']

	# get POS matrices for q1 and q2
	question1_POS_mat = np.array( [POS_dict[pos] for pos in pos_to_list(row['POS1']) ] )
	question2_POS_mat = np.array( [POS_dict[pos] for pos in pos_to_list(row['POS2']) ] )
	POS_mins = np.vstack((question1_POS_mat, question2_POS_mat))

	#get the min cos block distances for each word in q1 and q2
	inter_cos_dist_mat = two_question_distances(q1, q2, word_vectors, metric = 'cosine')
	q1_cos_mins_in_q2, q2_cos_mins_in_q1 = row_and_col_mins(inter_cos_dist_mat)
	inter_cos_mins = np.vstack((q1_cos_mins_in_q2, q2_cos_mins_in_q1))

	#get the min city block distances for each word in q1 and q2
	inter_cb_dist_mat = two_question_distances(q1, q2, word_vectors, metric = 'cityblock')
	q1_cb_mins_in_q2, q2_cb_mins_in_q1 = row_and_col_mins(inter_cb_dist_mat)
	inter_cb_mins = np.vstack((q1_cb_mins_in_q2, q2_cb_mins_in_q1))

	# get cosine mins within a single question (excluding 0s)
	q1_only_cos_matrix = one_question_distances(q1, word_vectors, metric = 'cosine')
	q2_only_cos_matrix = one_question_distances(q2, word_vectors, metric = 'cosine')
	q1_only_cos_mins = row_and_col_mins(q1_only_cos_matrix, nonzero_only = True)[0]
	q2_only_cos_mins = row_and_col_mins(q2_only_cos_matrix, nonzero_only = True)[0]
	intra_cos_mins = np.vstack((q1_only_cos_mins, q2_only_cos_mins))

	# get cityblock mins within a single question (excluding 0s)
	q1_only_cb_matrix = one_question_distances(q1, word_vectors, metric = 'cityblock')
	q2_only_cb_matrix = one_question_distances(q2, word_vectors, metric = 'cityblock')
	q1_only_cb_mins = row_and_col_mins(q1_only_cb_matrix, nonzero_only = True)[0]
	q2_only_cb_mins = row_and_col_mins(q2_only_cb_matrix, nonzero_only = True)[0]
	intra_cb_mins = np.vstack((q1_only_cb_mins, q2_only_cb_mins))

	return np.transpose(np.hstack((POS_mins, inter_cos_mins, inter_cb_mins, intra_cos_mins, intra_cb_mins)))



with open(filename) as datafile:
	reader = DictReader(datafile)
	for row in reader:
		input_matrix = assemble_row_inputs(row)
		#print("input mat shape is {}".format(input_matrix.shape))
		break #break for testing purposes
