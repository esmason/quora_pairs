import numpy as np
from scipy.spatial import distance
from sklearn.metrics.pairwise import pairwise_distances
from word2vec_loader import load_word_vectors

def two_question_distances(question1, question2, word_vectors, metric='cosine'):
	'''returns a distance matrix of size = [words in question1, words in question2] '''

	if not (type(question1) is str and type(question2) is str):
		raise AssertionError("question arguments must both be strings but they are {} and {}".format(type(question), type(question2)))
	word_array1 = [w for w in question1.split(" ") if len(w) > 0]
	word_array2 = [w for w in question2.split(" ") if len(w) > 0]
	if len(word_array1) == 0 or len(word_array2) == 0:
		raise AssertionError("a question cannot be an empty string")
	question_vec1 = word_vectors.__getitem__(word_array1)
	question_vec2 = word_vectors.__getitem__(word_array2)
	distance_matrix = pairwise_distances(question_vec1, question_vec2, metric=metric)
	return distance_matrix

def word_question_distances(word, question, word_vectors, metric='cosine'):
	'''returns a distance matrix of size = [1, words in question]'''
	return two_question_distances(word, question, word_vectors, metric = metric)

def one_question_distances(question, word_vectors,  metric='cosine'):
	'''if one question provided, returns distance matrix for all words in question
	   if 2 questions provided, returns distance for all inter-question combiations'''
	if not type(question) is str:
		raise AssertionError("question argument must be a string but it is {}".format(type(question)))
	word_array = [w for w in question.split(" ") if len(w) > 0]
	if len(word_array) == 0:
		raise AssertionError("question cannot be an empty string")
	question_vec = word_vectors.__getitem__(word_array)
	distance_matrix = pairwise_distances(question_vec, metric=metric)
	return distance_matrix

def min_distance(word, question, word_vectors, metric='cosine'):
	return np.amin(word_question_distances(word, question, word_vectors, metric = metric))

def row_and_col_mins(distance_matrix, nonzero_only = False):
	'''returns a list of two numpy col vectors, the first vector is the min distances for the question1, the second for question 2'''
	if nonzero_only:
		#make all zeros into ones
		distance_matrix += np.multiply(np.equal(distance_matrix, 0), np.ones(distance_matrix.shape))
	question1_array = np.amin(distance_matrix, axis = 1)
	question2_array = np.amin(distance_matrix, axis = 0)
	return [ question1_array.reshape((question1_array.shape[0], 1)),
			 question2_array.reshape((question2_array.shape[0], 1)) ]

def get_POS_dict():
	I_mat = np.eye(39)
	POS = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR',
		'JJS', 'LS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS',
		'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS',
		'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN',
		'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB', 'PERSON'
		'ORGANIZATION', 'LOCATION']
	return dict(zip(POS, [row for row in I_mat]))

def pos_to_list(pos):
	'''takes a string of POS and returns a list of string'''
	return [p for p in pos.split(" ") if len(p) > 0]
