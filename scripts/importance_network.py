import tensorflow as tf
from word2vec_utils import *
import csv
import numpy as np
from csv import DictReader
from word2vec_loader import load_word_vectors
import time
import pandas as pd
import multiprocessing as mp
from data_utils import load_csv

filename =  "../data/dataPos.csv"

#dist_mat_test = np.array([[0,1],[2,3]])
#print(row_and_col_mins(dist_mat_test))
word_vectors = load_word_vectors()
POS_dict = get_POS_dict()

def trim_q(q, pos_list):
	q_arr = q.split(" ")
	if len(q_arr) > 30:
		q_arr = q_arr[:30]
		pos_list = pos_list[:30]
	trimmed_q = " ".join(q_arr)
	return trimmed_q, pos_list

def assemble_row_inputs(row):
	'''returns a matrix where each column is a word in order [ex1_q1 words, ex1_q2 words, ex2_q1 words...]
		each row is a feature from top to bottom POS/NER tag(40), inter min cos distance(1),
		inter min city-block distance(1), intra min cos distance (nonzero) (1), intra min
		city block distance (nonzero only) (1), position in question [0-1] (1), word2vec vector (300),
		is_duplicate/ y vector (1), pair id (1)
		throws: AssertionError if one of the questions has no words in word2vec'''

	q1 = row['question1']
	q2 = row['question2']

	#truncate q1/ q2 at 30 words
	q1, pos_list1 = trim_q(q1, pos_to_list(row['POS1']))
	q2, pos_list2 = trim_q(q2, pos_to_list(row['POS2']))



	try:
		q1_word2vec = question_word2vec(q1, word_vectors)
		q2_word2vec = question_word2vec(q2, word_vectors)
		both_word2vec = np.vstack((q1_word2vec, q2_word2vec))
	except AssertionError:
		return None
	#print("shape of both word2vec is {}".format(both_word2vec.shape))

	#see if the word2vec are in the dictionary


	# get POS matrices for q1 and q2
	question1_POS_mat = np.array( [POS_dict[pos] for pos in pos_list1 ] )
	question2_POS_mat = np.array( [POS_dict[pos] for pos in pos_list2 ] )
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

	question_ids = np.vstack((np.ones((q1_cos_mins_in_q2.shape[0], 1))*int(row['qid1']) ,
							np.ones((q2_cos_mins_in_q1.shape[0], 1))*int(row['qid2']) ))

	pair_id = np.ones((word_positions.shape[0],1))*int(row['id'])

	try:
		ret_mat = np.transpose(np.hstack((POS_mins, #NEW used to be 39
								inter_cos_mins,
								inter_cb_mins,
								intra_cos_mins,
								intra_cb_mins,
								word_positions,
								both_word2vec,
								is_duplicate,
								question_ids, #NEW
								pair_id))).astype('float32')
	except ValueError:
		print(q1)
		print(q2)
		raise ValueError()
	return ret_mat

def idx_vec(counter, words):
	if words >= 30:
		q_idx = np.linspace(counter, counter + 30 - 1, 30, dtype='float32').reshape(1, 30)
		counter += 30
	else:
		q_idx = np.hstack( (np.linspace(counter, counter + words-1, words, dtype='float32').reshape(1,words) ,
		 					np.zeros(( 1, 30-words), dtype='float32')))
		counter += words
	return (counter, q_idx)


def process_chunk(sub_frame):
	#print("starting sub frame")
	counter = 1
	word_list = []
	first_id = 0
	input_matrix = np.zeros((348, 1))
	q1_idx = np.zeros((1, 30))
	q2_idx = np.zeros((1, 30))
	for i, row in sub_frame.iterrows():
		if first_id == 0:
			first_id = row['id']
			first_question = row['question1']
		q1_words = len(row['question1'].split(" "))
		q2_words = len(row['question2'].split(" "))
		#handle POS bug
		if  (not (len(row['POS1'].split(" ")) == q1_words) or not (len(row['POS2'].split(" ")) == q2_words)):
			#print("skipping example because of pos mis-label")
			continue
		row_input = assemble_row_inputs(row)
		if type(row_input) == type(None):
			#check to see if invalid question (ie no words in dictionary, or length of question is 0)
			continue

		input_matrix = np.hstack((input_matrix, row_input))
		counter, q1_row_idx = idx_vec(counter, q1_words)
		counter, q2_row_idx = idx_vec(counter, q2_words)
		q1_idx = np.vstack((q1_idx, q1_row_idx))
		q2_idx = np.vstack((q2_idx, q2_row_idx))

		if counter%10000 == 0:
			print(counter)
	#input_matrix.tofile('sub_matrix2_{}.bin'.format(first_id))
	#q1_idx.tofile('sub_q1_{}.bin'.format(first_id))
	#q2_idx.tofile('sub_q2_{}.bin'.format(first_id))
	return (input_matrix, q1_idx[1:,:], q2_idx[1:,:])
		#print("input_matrix shape {}".format(input_matrix.shape))
	#print("input mat shape is {}".format(input_matrix.shape))

dataframe = load_csv(filename)
#print("df shape {}".format(dataframe.shape))
num_processors = mp.cpu_count()
subframe = np.array_split(dataframe, 6)[5]
chunks = np.array_split(subframe, num_processors)
# create our pool with `num_processes` processes
pool = mp.Pool(processes = num_processors)
# apply our function to each chunk in the list
listResults = pool.map(process_chunk, chunks)

counter = 0
mat_list = []
q1_list = []
q2_list = []
for result in listResults:
	sub_mat, sub_q1_idx, sub_q2_idx = result
	sub_q1_idx += np.multiply(np.ones((sub_q1_idx.shape))*counter,  np.not_equal(sub_q1_idx, 0) )
	sub_q2_idx += np.multiply(np.ones((sub_q1_idx.shape))*counter, np.not_equal(sub_q2_idx, 0))
	counter = np.amax(sub_q2_idx) + 1
	mat_list.append(sub_mat)
	q1_list.append(sub_q1_idx)
	q2_list.append(sub_q2_idx)
final_matrix = np.hstack(mat_list).astype('float32')
final_q1 = np.vstack(q1_list).astype('float32')
final_q2 = np.vstack(q2_list).astype('float32')


print(final_q1.shape)
print(final_q2.shape)


final_matrix.tofile("mat_final_6.bin")
final_q1.tofile("q1_final_6.bin")
final_q2.tofile("q2_final_6.bin")

print(np.fromfile("mat_final_6.bin", dtype='float32').shape)
#print(np.load("input_matrix.csv")[39:,2])
