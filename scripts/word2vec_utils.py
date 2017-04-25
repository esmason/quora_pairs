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

def single_question_distances(question, word_vectors,  metric='cosine'):
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



#TESTS
word_vectors = load_word_vectors()
print("two question")
print(two_question_distances("banana greetings", "hello", word_vectors))
	
print("one question")
print(single_question_distances("hello tyrosine dopamine" , word_vectors))

print("word and question")
print(word_question_distances("hat", "toque scarf sock", word_vectors))
print(min_distance("hat", "toque scarf sock", word_vectors))


