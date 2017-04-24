from gensim.models.keyedvectors import KeyedVectors


def load_word_vectors():
	print("loading word vectors")
	word_vectors = KeyedVectors.load_word2vec_format('../bin/GoogleNews-vectors-negative300.bin', binary=True)
	return word_vectors
	
	


