import tensorflow as tf
from word2vec_utils import *
import csv
import numpy as np
from csv import DictReader
from word2vec_loader import load_word_vectors
import time
import pandas as pd
import multiprocessing as mp

filename = "../data/dataPos.csv"

#dist_mat_test = np.array([[0,1],[2,3]])
#print(row_and_col_mins(dist_mat_test))
word_vectors = load_word_vectors()
POS_dict = get_POS_dict()

def assemble_row_inputs(row):
	'''returns a matrix where each column is a word in order [ex1_q1 words, ex1_q2 words, ex2_q1 words...]
		each row is a feature from top to bottom POS/NER tag(39), inter min cos distance(1),
		inter min city-block distance(1), intra min cos distance (nonzero) (1), intra min
		city block distance (nonzero only) (1), position in question [0-1] (1), word2vec vector (300),
		is_duplicate/ y vector (1), pair id (1)'''

	q1 = row['question1']
	q2 = row['question2']

	q1_word2vec = question_word2vec(q1, word_vectors)
	q2_word2vec = question_word2vec(q2, word_vectors)
	both_word2vec = np.vstack((q1_word2vec, q2_word2vec))
	#print("shape of both word2vec is {}".format(both_word2vec.shape))

	# get POS matrices for q1 and q2
	question1_POS_mat = np.array( [POS_dict[pos] for pos in pos_to_list(row['POS1']) ] )
	question2_POS_mat = np.array( [POS_dict[pos] for pos in pos_to_list(row['POS2']) ] )
	POS_mins = np.vstack((question1_POS_mat, question2_POS_mat))
	#print("POS mins dims :{}".format(POS_mins.shape))


	#get the min cos block distances for each word in q1 and q2
	try:
		inter_cos_dist_mat = two_question_distances(q1, q2, word_vectors, metric = 'cosine')
		q1_cos_mins_in_q2, q2_cos_mins_in_q1 = row_and_col_mins(inter_cos_dist_mat)
	except KeyError:
		q1_cos_mins_in_q2, q2_cos_mins_in_q1 = two_question_sequential_mins(q1, q2, word_vectors, metric = 'cosine')
	inter_cos_mins = np.vstack((q1_cos_mins_in_q2, q2_cos_mins_in_q1))
	#print("inter cos mins dims :{}".format(inter_cos_mins.shape))



	#get the min city block distances for each word in q1 and q2
	try:
		inter_cb_dist_mat = two_question_distances(q1, q2, word_vectors, metric = 'euclidean')
		q1_cb_mins_in_q2, q2_cb_mins_in_q1 = row_and_col_mins(inter_cb_dist_mat)
	except KeyError:
		q1_cb_mins_in_q2, q2_cb_mins_in_q1 = two_question_sequential_mins(q1, q2, word_vectors, metric = 'euclidean')
	inter_cb_mins = np.vstack((q1_cb_mins_in_q2, q2_cb_mins_in_q1))
	#print("inter cb mins dims :{}".format(inter_cb_mins.shape))



	# get cosine mins within a single question (excluding 0s)
	try:
		q1_only_cos_matrix = one_question_distances(q1, word_vectors, metric = 'cosine')
		q1_only_cos_mins = row_and_col_mins(q1_only_cos_matrix, nonzero_only = True)[0]
	except KeyError:
		q1_only_cos_mins = one_question_sequential_mins(q1, word_vectors, metric = 'cosine')
	try:
		q2_only_cos_matrix = one_question_distances(q2, word_vectors, metric = 'cosine')
		q2_only_cos_mins = row_and_col_mins(q2_only_cos_matrix, nonzero_only = True)[0]
	except KeyError:
		q2_only_cos_mins = one_question_sequential_mins(q2, word_vectors, metric = 'cosine')
	#print("intra cos mins dims q1:{} q2{}".format(q1_only_cos_mins.shape, q2_only_cos_mins.shape))
	intra_cos_mins = np.vstack((q1_only_cos_mins, q2_only_cos_mins))


	# get cityblock mins within a single question (excluding 0s)
	try:
		q1_only_cb_matrix = one_question_distances(q1, word_vectors, metric = 'euclidean')
		q1_only_cb_mins = row_and_col_mins(q1_only_cb_matrix, nonzero_only = True)[0]
	except KeyError:
		q1_only_cb_mins = one_question_sequential_mins(q1, word_vectors, metric = 'euclidean')
	try:
		q2_only_cb_matrix = one_question_distances(q2, word_vectors, metric = 'euclidean')
		q2_only_cb_mins = row_and_col_mins(q2_only_cb_matrix, nonzero_only = True)[0]
	except KeyError:
		q2_only_cb_mins = one_question_sequential_mins(q2, word_vectors, metric = 'euclidean')
	#print("intra cb mins dims q1:{} q2{}".format(q1_only_cb_mins.shape, q2_only_cb_mins.shape))

	intra_cb_mins = np.vstack((q1_only_cb_mins, q2_only_cb_mins))

	word_positions = np.vstack((word_position_vector(q1), word_position_vector(q2)))

	#print("shape of word positions matrix is {}".format(word_positions.shape))

	is_duplicate = np.ones((word_positions.shape[0],1))*int(row['is_duplicate'])

	pair_id = np.ones((word_positions.shape[0],1))*int(row['id'])

	return np.transpose(np.hstack((POS_mins,
								inter_cos_mins,
								inter_cb_mins,
								intra_cos_mins,
								intra_cb_mins,
								word_positions,
								both_word2vec,
								is_duplicate,
								pair_id)))



def process_chunk(sub_frame):
	counter = 0
	for i, row in sub_frame.iterrows():
		counter +=1
		#handle POS bug
		if not len(row['POS1'].split(" ")) == len(row['question1'].split(" ")) or not (len(row['POS2'].split(" ")) == len(row['question2'].split(" "))):
			#print("skipping example because of pos mis-label")
			continue
		try:
			input_matrix = np.hstack((input_matrix, assemble_row_inputs(row)))
		except:
			#for first run
			input_matrix = assemble_row_inputs(row)
		if counter%3000 == 0:
			print(counter)
	return input_matrix
		#print("input_matrix shape {}".format(input_matrix.shape))
	#print("input mat shape is {}".format(input_matrix.shape))

dataframe = np.array_split(pd.read_csv(filename), 10)[0]
print("df shape {}".format(dataframe.shape))
num_processors = mp.cpu_count()
chunks = np.array_split(dataframe, num_processors)
# create our pool with `num_processes` processes
pool = mp.Pool(processes = num_processors)
# apply our function to each chunk in the list
listResults = pool.map(process_chunk, chunks)

# puts everything back together and puts back t oold sorting
for result in listResults:
	print(result.shape)
final_matrix = np.hstack(listResults)
print(final_matrix.shape)

final_matrix.tofile("name.bin")
print(np.fromfile("name.bin").reshape(final_matrix.shape))
print(np.fromfile("name.bin").reshape(final_matrix.shape).shape)
#print(np.load("input_matrix.csv")[39:,2])
