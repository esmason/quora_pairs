import numpy as np
from scipy.spatial import distance
from sklearn.metrics.pairwise import pairwise_distances
from word2vec_loader import load_word_vectors


def one_question_sequential_mins(q1, word_vectors, metric = 'cosine'):
	array = [w for w in q1.split(" ") if len(w) > 0]
	question_vec = []
	missed_words = []
	for i, word in enumerate(array):
		try:
			vector = word_vectors.__getitem__(word)
			question_vec.append(vector)
		except KeyError:
			missed_words.append((i, word))
	#compute mins of words not including non-word2vec words
	distance_matrix = pairwise_distances(question_vec, metric=metric)
	dict_mins = row_and_col_mins(distance_matrix, nonzero_only = True)[0]
	#just use average of other mins
	average_min = np.mean(dict_mins)
	for mw in missed_words:
		dict_mins = np.insert(dict_mins, mw[0], average_min)
	return dict_mins.reshape((dict_mins.shape[0], 1))

def question_word2vec(q1, word_vectors):
	'''returns the word vectors for the question and if a word is not in the word_vectors
	dictionary, in it's place return the mean of the other word vectors in the question'''
	words_in_dict = 0
	try:
		array = [w for w in q1.split(" ") if len(w) > 0]
	except AttributeError:
		print("non-string {}".format(q1) )
		raise AttributeError
	question_vec = []
	for i, word in enumerate(array):
		try:
			vector = word_vectors.__getitem__(word).reshape((1, 300))
			#print("vector shape: {}".format(vector.shape))
			question_vec.append(vector)
			try:
				array_for_mean = np.vstack((array_for_mean, vector))
			except:
				array_for_mean = vector
			#print("array_for_mean shape: {}".format(array_for_mean.shape))
			words_in_dict += 1
		except KeyError:
			if word.isnumeric():
				question_vec.append(word_vectors.__getitem__('number'))
			else:
				question_vec.append(None)
	if words_in_dict == 0:
		print("no words found for q: {}".format(q1))
		raise AssertionError()
	mean_vec = np.mean(array_for_mean, axis=0).reshape((1, 300))
	#print("mean vec shape is {}".format(mean_vec.shape))
	#initialize return_vec to first element of question_vec (first word vec in q)
	return_vec = question_vec.pop(0)
	if type(return_vec)== type(None): ###
		return_vec = mean_vec
	for word_vec in question_vec:
		if  type(word_vec) == type(None): ###
			word_vec = mean_vec
		return_vec = np.vstack((return_vec, word_vec))
	#print("return vec shape is {}".format(return_vec.shape))
	return return_vec




def word_position_vector(question):
	words = len([w for w in question.split(" ") if len(w)>0])
	return np.linspace(0, 1, words).reshape((words, 1))


def two_question_sequential_mins(q1, q2, word_vectors, metric = 'cosine'):
	'''this function should only be used if word_vectors does not contain a specific
	question word'''
	array1 = [w for w in q1.split(" ") if len(w) > 0]
	array2 = [w for w in q2.split(" ") if len(w) > 0]
	if len(array1) == 0 or len(array2) == 0:
		raise AssertionError("question cannot be an empty string")
	#first determine which words are not in word vectors and create the word vectors without those
	#words
	question_vec1 = []
	missed_words_1 = []
	for i, word in enumerate(array1):
		try:
			vector = word_vectors.__getitem__(word)
			question_vec1.append(vector)
		except KeyError:
			missed_words_1.append((i, word))

	question_vec2 = []
	missed_words_2 = []
	for i, word in enumerate(array2):
		try:
			vector = word_vectors.__getitem__(word)
			question_vec2.append(vector)
		except KeyError:
			missed_words_2.append((i, word))
	#compute word2vec dict mins
	distance_matrix = pairwise_distances(question_vec1, question_vec2, metric = metric)
	dict_mins1, dict_mins2 = row_and_col_mins(distance_matrix)
	average_min1 = np.mean(dict_mins1)
	average_min2 = np.mean(dict_mins2)
	for mw1 in missed_words_1:
		#check to see if that exact word is in the other question
		if mw1[1] in q2:
			dict_mins1 = np.insert(dict_mins1, mw1[0], 0.0)
		else:
			dict_mins1 = np.insert(dict_mins1, mw1[0], average_min1)
	for mw2 in missed_words_2:
		#check to see if that exact word is in the other question
		if mw2[1] in q1:
			dict_mins2 = np.insert(dict_mins2, mw2[0], 0.0)
		else:
			dict_mins2 = np.insert(dict_mins2, mw2[0], average_min2)
	return [dict_mins1.reshape((dict_mins1.shape[0], 1)), dict_mins2.reshape((dict_mins2.shape[0], 1)) ]


def two_question_distances(question1, question2, word_vectors, metric='cosine'):
	'''returns a distance matrix of size = [words in question1, words in question2]
		throws: KeyError if question contains words not in word_vectors provided'''

	if not (type(question1) is str and type(question2) is str):
		raise AssertionError("question arguments must both be strings but they are {} and {}".format(type(question), type(question2)))
	word_array1 = [w for w in question1.split(" ") if len(w) > 0]
	word_array2 = [w for w in question2.split(" ") if len(w) > 0]
	if len(word_array1) == 0 or len(word_array2) == 0:
		raise AssertionError("a question cannot be an empty string")
	#try to process all words simultaneously
	question_vec1 = word_vectors.__getitem__(word_array1)
	question_vec2 = word_vectors.__getitem__(word_array2)
	distance_matrix = pairwise_distances(question_vec1, question_vec2, metric=metric)
	return distance_matrix

def word_question_distances(word, question, word_vectors, metric='cosine'):
	'''returns a distance matrix of size = [1, words in question]'''
	return two_question_distances(word, question, word_vectors, metric = metric)

def one_question_distances(question, word_vectors,  metric='cosine'):
	'''if one question provided, returns distance matrix for all words in question
	   if 2 questions provided, returns distance for all inter-question combiations
	   throws: KeyError if question contains words not in word_vectors provided'''
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
	I_mat = np.eye(40)
	POS = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR',
		'JJS', 'LS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS',
		'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS',
		'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN',
		'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB', 'PERSON',
		'ORGANIZATION', 'LOCATION', '$']
	return dict(zip(POS, [row for row in I_mat]))

def pos_to_list(pos):
	'''takes a string of POS and returns a list of string'''
	return [p for p in pos.split(" ") if len(p) > 0]
