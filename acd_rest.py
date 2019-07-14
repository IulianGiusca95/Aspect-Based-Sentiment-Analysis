import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras import layers
from sklearn.preprocessing import OneHotEncoder
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import xml.etree.ElementTree as ET, getopt, logging, sys, random, re, copy, os
from lxml import etree
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV

from keras import backend as K

def getSentences(file):
  tree = ET.parse(file, etree.XMLParser(recover=True, encoding="utf-8"))
  root = tree.getroot()
  s = []
  p = []
  for review in root.findall('Review'):
    sent = []
    sent_characteristics = []
    for sentences in review.findall('sentences'):
      for sentence in sentences.findall('sentence'):
        text = sentence.find('text').text
        sent.append(text)
        polarity = []
        for opinions in sentence.findall('Opinions'):
          for opinion in opinions.findall('Opinion'):
            elem = [opinion.get('category'), opinion.get('polarity'), opinion.get('target'), opinion.get('from'), opinion.get('to')]
            polarity.append(elem)
        sent_characteristics.append(polarity)
    s.append(sent)
    p.append(sent_characteristics)
        
  return s, p
  
train_sentences, train_adnotations = getSentences("data\\restaurants\\train.xml")
test_sentences, test_adnotations = getSentences("data\\restaurants\\test.xml")

train_reviews = []
train_aspects = []
test_reviews = []
test_aspects = []

for review in train_sentences:
  train_reviews.append(' '.join(review))
    
for ta in train_adnotations:
  aspect = set()
  for adnotation_set in ta:
    for a in adnotation_set:
      aspect.add(a[0])
  train_aspects.append(aspect)

for review in test_sentences:
  test_reviews.append(' '.join(review))
  
for ta in test_adnotations:
  aspect = set()
  for adnotation_set in ta:
    for a in adnotation_set:
      aspect.add(a[0])
  test_aspects.append(aspect)

def getLabels(aspects):
	#print(unique_aspects)
	#Create train labels
	restaurant_general = []
	food_quality = []
	restaurant_misc = []
	food_prices = []
	drinks_quality = []
	location_general = []
	restaurant_prices = []
	ambience_general = []
	drinks_style_options = []
	service_general = []
	drinks_prices = []
	food_style_options = []

	for aspect in aspects:
		if 'RESTAURANT#GENERAL' in aspect:
			restaurant_general.append(1)
		else:
			restaurant_general.append(0)
			
		if 'FOOD#QUALITY' in aspect:
			food_quality.append(1)
		else:
			food_quality.append(0)
			
		if 'RESTAURANT#MISCELLANEOUS' in aspect:
			restaurant_misc.append(1)
		else:
			restaurant_misc.append(0)
			
		if 'FOOD#PRICES' in aspect:
			food_prices.append(1)
		else:
			food_prices.append(0)
			
		if 'DRINKS#QUALITY' in aspect:
			drinks_quality.append(1)
		else:
			drinks_quality.append(0)
			
		if 'LOCATION#GENERAL' in aspect:
			location_general.append(1)
		else:
			location_general.append(0)
			
		if 'RESTAURANT#PRICES' in aspect:
			restaurant_prices.append(1)
		else:
			restaurant_prices.append(0)
			
		if 'AMBIENCE#GENERAL' in aspect:
			ambience_general.append(1)
		else:
			ambience_general.append(0)
			
		if 'DRINKS#STYLE_OPTIONS' in aspect:
			drinks_style_options.append(1)
		else:
			drinks_style_options.append(0)
			
		if 'SERVICE#GENERAL' in aspect:
			service_general.append(1)
		else:
			service_general.append(0)
			
		if 'DRINKS#PRICES' in aspect:
			drinks_prices.append(1)
		else:
			drinks_prices.append(0)
			
		if 'FOOD#STYLE_OPTIONS' in aspect:
			food_style_options.append(1)
		else:
			food_style_options.append(0)
				
	return restaurant_general, food_quality, restaurant_misc, food_prices, drinks_quality, location_general, restaurant_prices, ambience_general, drinks_style_options,service_general, drinks_prices, food_style_options
	
#Test and Train labels
test1, test2, test3, test4, test5, test6, test7, test8, test9, test10, test11, test12 = getLabels(test_aspects)
test_labels = [test1, test2, test3, test4, test5, test6, test7, test8, test9, test10, test11, test12]

train1, train2, train3, train4, train5, train6, train7, train8, train9, train10, train11, train12 = getLabels(train_aspects)
train_labels = [train1, train2, train3, train4, train5, train6, train7, train8, train9, train10, train11, train12]

#Vectorizing data
vectorizer = CountVectorizer(analyzer='word', lowercase=True, stop_words='english', ngram_range=(1,2))
vectorizer.fit(train_reviews)

x_train = vectorizer.transform(train_reviews)
x_test = vectorizer.transform(test_reviews)

input_dim = x_train.shape[1]

tokenizer = Tokenizer(num_words = 5000)
tokenizer.fit_on_texts(train_reviews)

x_train = tokenizer.texts_to_sequences(train_reviews)
x_test = tokenizer.texts_to_sequences(test_reviews)
vocab_size = len(tokenizer.word_index) + 1 

maxlen = 100
x_train = pad_sequences(x_train, padding='post', maxlen=maxlen)
x_test = pad_sequences(x_test, padding = 'post', maxlen=maxlen)

#Pretrained Word Embeddings
def create_embedding_matrix(filepath, word_index, embedding_dim):
	vocab_size = len(word_index) + 1
	embedding_matrix = np.zeros((vocab_size, embedding_dim))
	
	with open(filepath, encoding='utf-8') as f:
		for line in f:
			word, *vector = line.split()
			if word in word_index:
				idx = word_index[word]
				embedding_matrix[idx] = np.array(vector, dtype=np.float32)[:embedding_dim]

	return embedding_matrix

embedding_dim = 200
embedding_matrix = create_embedding_matrix('data\\glove.6B.200d.txt', tokenizer.word_index, embedding_dim)

nonzero_elements = np.count_nonzero(np.count_nonzero(embedding_matrix, axis = 1))
#print(nonzero_elements / vocab_size)

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def getPredictions(x_train, x_test, train, test):
	embedding_dim = 200
	embedding_matrix = create_embedding_matrix('data\\glove.6B.200d.txt', tokenizer.word_index, embedding_dim)
	model = Sequential()
	model.add(layers.Embedding(vocab_size, embedding_dim, weights=[embedding_matrix], input_length = maxlen, trainable = True))
	model.add(layers.Conv1D(64, 3, activation = 'relu'))
	model.add(layers.GlobalMaxPool1D())
	model.add(layers.Dense(10, activation='relu'))
	model.add(layers.Dense(1, activation='sigmoid'))
	model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[f1, 'accuracy'])
	#model.summary()

	history = model.fit(x_train, train, epochs = 20, verbose = False, validation_data = (x_test, test), batch_size = 10)
	val = model.evaluate(x_train, train, verbose = False)
	val = model.evaluate(x_test, test, verbose = False)

	predictions = model.predict(x_test)
	return predictions
	
print("Getting Predictions1")
predictions1 = getPredictions(x_train, x_test, train1, test1)
print("Getting Predictions2")
predictions2 = getPredictions(x_train, x_test, train2, test2)
print("Getting Predictions3")
predictions3 = getPredictions(x_train, x_test, train3, test3)
print("Getting Predictions4")
predictions4 = getPredictions(x_train, x_test, train4, test4)
print("Getting Predictions5")
predictions5 = getPredictions(x_train, x_test, train5, test5)
print("Getting Predictions6")
predictions6 = getPredictions(x_train, x_test, train6, test6)
print("Getting Predictions7")
predictions7 = getPredictions(x_train, x_test, train7, test7)
print("Getting Predictions8")
predictions8 = getPredictions(x_train, x_test, train8, test8)
print("Getting Predictions9")
predictions9 = getPredictions(x_train, x_test, train9, test9)
print("Getting Predictions10")
predictions10 = getPredictions(x_train, x_test, train10, test10)
print("Getting Predictions11")
predictions11 = getPredictions(x_train, x_test, train11, test11)
print("Getting Predictions12")
predictions12 = getPredictions(x_train, x_test, train12, test12)

predicted_aspects = []
nr = len(test_sentences)

for i in range(nr):
  predicted_aspect = []
  
  if predictions1[i] > 0.4:
    predicted_aspect.append('RESTAURANT#GENERAL')
  if predictions2[i] > 0.4:
    predicted_aspect.append('FOOD#QUALITY')
  if predictions3[i] > 0.4:
    predicted_aspect.append('RESTAURANT#MISCELLANEOUS')
  if predictions4[i] > 0.4:
    predicted_aspect.append('FOOD#PRICES')
  if predictions5[i] > 0.4:
    predicted_aspect.append('DRINKS#QUALITY')
  if predictions6[i] > 0.4:
    predicted_aspect.append('LOCATION#GENERAL')
  if predictions7[i] > 0.4:
    predicted_aspect.append('RESTAURANT#PRICES')
  if predictions8[i] > 0.4:
    predicted_aspect.append('AMBIENCE#GENERAL')
  if predictions9[i] > 0.4:
    predicted_aspect.append('DRINKS#STYLE_OPTIONS')
  if predictions10[i] > 0.4:
    predicted_aspect.append('SERVICE#GENERAL')
  if predictions11[i] > 0.4:
    predicted_aspect.append('DRINKS#PRICES')
  if predictions12[i] > 0.4:
    predicted_aspect.append('FOOD#STYLE_OPTIONS')
    
  predicted_aspects.append(predicted_aspect)
  
#Evaluating the system
common_aspects = 0
relevant_aspects = 0
retrieved_aspects = 0

for i in range(nr):
  correct = set()
  for aspect in test_aspects[i]:
    correct.add(aspect)
  
  predicted = set()
  for aspect in predicted_aspects[i]:
    predicted.add(aspect)
    
  relevant_aspects = relevant_aspects + len(correct)
  retrieved_aspects = retrieved_aspects + len(predicted)
  common_aspects = common_aspects+len([aspect for aspect in predicted if aspect in correct])
  
print("Common aspects: ", common_aspects)
print("Retrieved aspects: ", retrieved_aspects)
print("Relevant aspects: ", relevant_aspects)
precision = common_aspects / retrieved_aspects if retrieved_aspects > 0 else 0
recall = common_aspects / relevant_aspects
f1_measure = 2 * precision * recall / (precision + recall)
print("Precision: ", precision)
print("Recall: ", recall)
print("F1: ", f1_measure)
