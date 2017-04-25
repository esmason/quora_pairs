import numpy as np
from word2vec_loader import load_word_vectors
from word2vec_utils import two_question_distances, one_question_distances, word_question_distances, min_distance
#TESTS


word_vectors = load_word_vectors()
print("two question")
print(two_question_distances("banana greetings", "hello", word_vectors))

print("one question")
print(one_question_distances("hello tyrosine dopamine" , word_vectors))


print("word and question, cosine")
print(word_question_distances("hat", "toque scarf sock", word_vectors, metric = 'cosine'))
print(min_distance("hat", "toque scarf sock", word_vectors, metric = 'cosine'))


print("word and question, cityblock")
print(word_question_distances("hat", "toque scarf sock", word_vectors, metric = 'cityblock'))
print(min_distance("hat", "toque scarf sock", word_vectors, metric = 'cityblock'))
