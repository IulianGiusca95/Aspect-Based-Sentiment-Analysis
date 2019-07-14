import pandas as pd
import numpy as np 
import spacy
nlp = spacy.load('en')
import csv
import xml.etree.ElementTree as ET, getopt, logging, sys, random, re, copy, os
from lxml import etree
import nltk
import numpy as np
import matplotlib.pyplot as plt
import os,sys,glob
import re
import sklearn
from bs4 import BeautifulSoup
from nltk import sent_tokenize
from nltk import pos_tag
from nltk import word_tokenize
from nltk.tokenize import WordPunctTokenizer
from textblob import TextBlob
from pprint import pprint
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, cross_val_score
from time import time
from scipy.sparse import coo_matrix, vstack, hstack
from sklearn.svm import LinearSVC
from sklearn import svm
from scipy import sparse
import matplotlib.pyplot as plt
from sklearn.preprocessing import Binarizer
import string


separator = ['and', ';', ',', '...', '(', 'so', 'yet', 'unless', ':', 'at', 'but', '-', 'with', 'though', 'although', '--', 'while', 'about', 'for', 'including', 'especially', 'in', '..', 'to', 'until', 'and_NOT', ';', ',', '...', '(', 'so_NOT', 'yet_NOT', 'unless_NOT', ':', 'at_NOT', 'but_NOT', '-', 'with_NOT', 'though_NOT', 'although_NOT', '--', 'while_NOT', 'about_NOT', 'for_NOT', 'including_NOT', 'especially_NOT', 'in_NOT', '..', 'to_NOT', 'until_NOT']
nouns = ['NN', 'NNS', 'NNP']
verbs = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
adverbs = ['RB', 'RBS', 'WRB', 'RBR']
adjectives = ['JJ', 'JJR', 'JJS']
global_brackets_fix = []

def getSentences(file):
	tree = ET.parse(file, etree.XMLParser(recover=True, encoding="utf-8"))
	root = tree.getroot()
	s = []
	p = []
	for review in root.findall('Review'):
		for sentences in review.findall('sentences'):
			for sentence in sentences.findall('sentence'):
				text = sentence.find('text').text
				s.append(text)
				polarity = []
				for opinions in sentence.findall('Opinions'):
					for opinion in opinions.findall('Opinion'):
						elem = [opinion.get('category'), opinion.get('polarity'), opinion.get('target'), opinion.get('from'), opinion.get('to')]
						polarity.append(elem)
				p.append(polarity)
				
	return s, p
 
def getIndices(sentence, attributes):
	indices = []
	marked_words = []
	words = nltk.word_tokenize(sentence)
	for a in attributes:
		if a[2] != 'NULL':
			wt = nltk.word_tokenize(a[2])
			ok = 0
			if wt[0] in words:
				indices.append(words.index(wt[0]))
				marked_words.append([words[words.index(wt[0])], words.index(wt[0])])
				ok = 1
				words[words.index(wt[0])] = words[words.index(wt[0])] + "_marked"
			elif wt[0]+"_NOT" in words:
				indices.append(words.index(wt[0]+"_NOT"))
				marked_words.append([words[words.index(wt[0]+"_NOT")], words.index(wt[0]+"_NOT")])
				ok = 1
				words[words.index(wt[0]+"_NOT")] = words[words.index(wt[0]+"_NOT")] + "_marked"
				
			if ok == 0:	#nu a gasit termen - il cauta in cuvintele deja marcate
				for mt in marked_words:
					if wt[0] == mt[0] or wt[0]+"_NOT" == mt[0]:
						indices.append(mt[1])		
	return indices
 
def fix_brackets(attr):

	if len(attr) != 5:
		for e in attr:
			fixed_attr = fix_brackets(e)

	else:
		global global_brackets_fix
		global_brackets_fix.append(attr)

def fix_sentences(sentence):

	if isinstance(sentence, str) is False: #Verific daca argumentul de input e o lsita sau un string
		for item in sentence:
			fix_sentences(item)
	else:
		global global_brackets_fix
		global_brackets_fix.append(sentence)
 
#Obtine lista de terms dintr-un review cu mai multe aspecte
def get_target(attributes):
	terms = []
	for attribute in attributes:
		terms.append(attribute[2])
	return terms

#Returneaza nr de targets nuli	
def number_of_null_targets(target):
	contor = 0
	for t in target:
		if t == 'NULL':
			contor = contor + 1
	return contor
	
def getSentenceSeparators(words):
	sep = []
	for i, w in enumerate(words):
		if w in separator:
			sep.append(i)
	return sep
	
def updateIndex(words, oldIndex, modifier):
	newIndex = []
	if modifier == 0: #Nu stergem elemente din index, ci doar actualizam valorile
		val = len(words)
		for ind in oldIndex:
			newIndex.append(ind - val)
	
	if modifier == 1: #Actualizam valorile si scoatem prima valoare din lista, deoarece eliminam primul termen cand facem split
		val = len(words)
		for ind in oldIndex:
			newIndex.append(ind-val)
		newIndex.remove(newIndex[0])
	
	if modifier == 2: #Doar stergem ultima valoare
		for ind in oldIndex:
			newIndex.append(ind)
		newIndex.remove(newIndex[len(newIndex)-1])
		
	if modifier == 3: #Lasam prima valoarea neschimbata, nu o adaugam pe a doua, modificam restul
		val = len(words)
		for i in range(len(oldIndex)):
			if i == 0:
				newIndex.append(oldIndex[i])
			if i >=2:
				if oldIndex[i] == oldIndex[0]:
					newIndex.append(oldIndex[i])
				else:
					newIndex.append(oldIndex[i]-val)
		

	
	return newIndex
	
def perform_split_all_null_targets(sentence, attributes):
	words = nltk.word_tokenize(sentence)
	if len(attributes) != 1: #Nr de splits = Nr de terms - 1
		sep = getSentenceSeparators(words)
		split_point = sep[len(sep)-1] #Luam ultimul separator
		if split_point == len(words)-1 and len(words)-2 >= 0:
			split_point = sep[len(sep)-2]
		sentence_chunk = words[split_point:] #Consideram toata propozitia de la ultimul separator pana la finalul ei
		attribute_chunk = attributes[len(attributes)-1] 
		attributes.remove(attribute_chunk) #Inlaturam ultimul atribut din lista de atribute
		
		remaining_sentence = " ".join(words[0:split_point])
		final_sentences, final_attributes = perform_split_all_null_targets(remaining_sentence, attributes) #Apelam functia recursiv pentru restul de separari
		final_sentences.append(" ".join(sentence_chunk))
		final_attributes.append(attribute_chunk)
	elif len(attributes) == 1:
		return [sentence], [attributes]
		
	return final_sentences, final_attributes
	
def perform_split_remaining_null_targets(L, index):
	sentence = L[0]
	attributes = L[1]
	final_sentences=[]
	final_attributes=[]
	words = nltk.word_tokenize(sentence)
	target = get_target(attributes)
	if number_of_null_targets(target) != 0:
		#Mai intai, incercam sa facem split la inceputul propozitiei
		chunked = 0
		first_target_index = index[0] #index-ul primului element target adnotat, din stanga propozitiei
		for i in range(1, first_target_index): #Ma uit la toate cuvintele dintre inceputul propozitiei si primul cuvant target
			if words[i] in separator:
				separator_index = i
				sentence_chunk = words[0:separator_index]
				attribute_chunk=[]
				for a in attributes:
					if a[2] == 'NULL':
						attribute_chunk = a
						attributes.remove(attribute_chunk)
						attribute_chunk[3] = -1 # - 1 marcheaza faptul ca a fost taiat de la inceputul frazei
						attribute_chunk[4] = -1
						break
				chunked = 1
				remaining_sentence = " ".join(words[separator_index:])
				index = updateIndex(sentence_chunk, index, 0)
				L1, final_index = perform_split_remaining_null_targets([remaining_sentence, attributes], index)
				L1_item = [" ".join(sentence_chunk), attribute_chunk]
				L1.append(L1_item)
				if chunked == 1:
					break
				
				
		if chunked == 0: #Daca nu am reusit sa efectuam split la inceput, incercam la finalul propozitiei
			last_target_index = index[len(index)-1] #index-ul ultimului element target adnotat, din dreapta propozitiei
			for i in range(len(words)-1, last_target_index, -1):
				if words[i] in separator:
					separator_index = i
					sentence_chunk = words[separator_index:]
					attribute_chunk=[]
					for a in attributes:
						if a[2] == 'NULL':
							attribute_chunk = a
							attributes.remove(attribute_chunk)
							attribute_chunk[3] = 500 # 500 marcheaza faptul ca a fost taiat de la sfarsitul frazei
							attribute_chunk[4] = 500

							break
					chunked = 1
					remaining_sentence = " ".join(words[0:separator_index])
					L1, final_index = perform_split_remaining_null_targets([remaining_sentence, attributes], index)

					L1_item = [" ".join(sentence_chunk), attribute_chunk]
					L1.append(L1_item)
					if chunked == 1:
						break
			
			
			if chunked == 0:
				if len(attributes) >= 3:
					for a in attributes:
						if a[2] == 'NULL':
							null_index = attributes.index(a)
					index_left = null_index-1
					left_found = -1
					while index_left >= 0:
						if attributes[index_left][2] != 'NULL':
							left_found = 1
							break
						index_left = index_left - 1
						
					index_right = null_index + 1
					right_found = -1
					while index_right < len(attributes):
						if attributes[index_right][2] != 'NULL':
							right_found = +1
							break
						index_right = index_right + 1
						
					#Am obtinut indecsii, acum sa aflam ce target words iconjoara propozitia null
					if left_found == 1 and right_found == 1:
						target_word_left = attributes[index_left][2]
						target_word_right = attributes[index_right][2]
						
						#Am obtinut indicii la care se gasesc cuvintele target
						index1 = words.index(target_word_left)
						index2 = words.index(target_word_right)
						
						#Cautam separatorii (la dreapta primului cuvant si la stanga celui de-al doilea cuvant)
						#care separa propozitia cu target null
						index_sep_1 = -1
						index_sep_2 = -1
						for i1 in range(index1, len(words)):
							if words[i1] in separator:

								index_sep_1 = i1
								break
								
						for i2 in range(index2, 0, -1):
							if words[i2] in separator:

								index_sep_2 = i2
								break
						
						if index_sep_1 != -1 and index_sep_2 != -1: #Am identificat propoizitia - o scoatem din text
							sentence_chunk= words[index_sep_1:index_sep_2]
							attribute_chunk = attributes[null_index]
							remaining_sentence_1 = " ".join(words[0:index_sep_1])
							remaining_sentence_2 = " ".join(words[index_sep_2:])
							remaining_sentence = remaining_sentence_1 + ' ' + remaining_sentence_2
							attributes.remove(attribute_chunk)
							index = updateIndex(sentence_chunk, index, 3)
							L1, final_index = perform_split_remaining_null_targets([remaining_sentence, attributes], index)
							L1_item = [" ".join(sentence_chunk), attribute_chunk]
							L1.append(L1_item)
							chunked = 1
			
			if chunked == 0: #Daca nu am reusit nici la inceput, nici la sfarsit, termenul neutru e la mijloc!
				return [[sentence, attributes]], index
	
	else: #Daca nu mai exista targets null 
		return [[sentence, attributes]], index
	
	
	return L1, final_index
		
def perform_split_on_targets(L, index):

	
	sentence = L[0]
	attributes = L[1]

	#print(index)
	final_sentences= []
	final_attributes = []
	final_index = []
	words = nltk.word_tokenize(sentence)
	
	if len(index) == 1: #cazul in care ramanem doar cu un singur termen; returnam tot
		global global_brackets_fix
		fix_brackets(attributes)
		attributes = global_brackets_fix[0]
		global_brackets_fix = []
		return [[sentence, attributes]], index
		

	
	elif index[0] == index[1]-1: #cazul in care avem doi termeni alaturati in stanga listei
		sentence_chunk = words[0:index[0]+1]
		attribute_chunk = attributes[0]
				
		attributes.remove(attribute_chunk)
		remaining_sentence = " ".join(words[index[0]+1:])
		index = updateIndex(sentence_chunk, index, 1)
		L1, final_index = perform_split_on_targets([remaining_sentence, attributes], index)
		L1_item = [" ".join(sentence_chunk), attribute_chunk]
		L1.append(L1_item)
		
	elif index[0] == index[1] and len(attributes) == 2:
		#Cautam separator
		best_separator_value = 100
		best_separator_index = -1
		for i in range(len(words)):
			if words[i] in separator:
				s1 = words[0:i]
				s2 = words[i:]
				if abs(len(s1)-len(s2)) < best_separator_value:
					best_separator_index=i
					best_separator_value = abs(len(s1)-len(s2))
		i = best_separator_index
		sentence_chunk = words[0:i]
		remaining_sentence = " ".join(words[i:])
		attribute_chunk = attributes[0]
		attributes.remove(attribute_chunk)
		index = updateIndex(sentence_chunk, index, 1)
		L1, final_index = perform_split_on_targets([remaining_sentence, attributes], index)
		L1_item = [" ".join(sentence_chunk), attribute_chunk]
		L1.append(L1_item)
		
				
		
	elif index[0] == index[1] and len(attributes)>2: #cazul in care acelasi termen defineste 2 aspecte

		#Cautam sa eliminam propozitia care nu contine termenul 
		first_target_index = index[0]
		found_sep = - 1
		for i in range(first_target_index, len(words)): #cautam la dreapta
			if words[i] in separator:
				found_sep = 1
				sentence_chunk = words[0:i]
				if words[index[0]] in sentence_chunk: #Daca prima propozitie continea termen, o eliminam pe a doua
					found_separator = -1
					second_index = i
					for j in range(i+1, len(words)):
						if words[j] in separator:
							found_separator = j
							break
					if found_separator == -1: #nu am gasit separator, ci am ajuns la finalul propozitiei, deci nu exista un al treilea termen conform regulilor
						sentence_chunk = words[j:len(words)-1]
						attribute_chunk = attributes[len(attributes)-1]
						remaining_sentence = " ".join(words[0:j])
						attributes.remove(attribute_chunk)
						index = updateIndex(sentence_chunk, index, 2)
						L1, final_index = perform_split_on_targets([remaining_sentence, attributes], index)
						L1_item = [" ".join(sentence_chunk), attribute_chunk]
						L1.append(L1_item)
					elif found_separator != -1 and len(index)==2: #Am gasit separator, dar suntem la finalul propozitiei
						sentence_chunk = words[i:len(words)-1]
						attribute_chunk = attributes[len(attributes)-1]
						remaining_sentence = " ".join(words[0:i])
						attributes.remove(attribute_chunk)
						index = updateIndex(sentence_chunk, index, 2)
						L1, final_index = perform_split_on_targets([remaining_sentence, attributes], index)
						L1_item = [" ".join(sentence_chunk), attribute_chunk]
						L1.append(L1_item)
					else: #Am gasit separator si nu suntem la finalul propozitiei
						if i+1 == j:
							continue
						sentence_chunk = words[i: j]
						attribute_chunk = attributes[1]
						remaining_sentence_1 = " ".join(words[0:i])
						remaining_sentence_2 = " ".join(words[j:])
						remaining_sentence = remaining_sentence_1 +' '+ remaining_sentence_2
						attributes.remove(attribute_chunk)
						index = updateIndex(sentence_chunk, index, 3)
						L1, final_index = perform_split_on_targets([remaining_sentence, attributes], index)
						L1_item = [" ".join(sentence_chunk), attribute_chunk]
						L1.append(L1_item)
				else: #Daca prima propozitie nu continea termen, atunci ea trebuie eliminata
					sentence_chunk = words[0:i]
					attribute_chunk = attributes[0]
					remaining_sentence = " ".join(words[i:len(words)-1])
					index = updateIndex(sentence_chunk, index, 1)
					attributes.remove(attribute_chunk)
					L1, final_index = perform_split_on_targets([remaining_sentence, attributes], index)
					L1_item = [" ".join(sentence_chunk), attribute_chunk]
					L1.append(L1_item)
				break
				
		if found_sep == -1: #Nu am gasit separator uitandu-ne la dreapta propozitiei, deci targetul se afla in ultima propozitie
			for i in range(first_target_index, 0, -1):
				found_sep2 = -1
				if words[i] in separator: #am gasit separatorul care delimiteaza ultima propozitie
					found_sep2 = 1
					second_index = words.index(words[i])
					#Cautam urmatorul separator, parcurgand lista de la dreapta la stanga, ptr a elimina penultima propozitie
					found_separator = -1
					for j in range(second_index, 0, -1):
						if words[j] in separator:
							found_separator = words.index(words[j])
							break
					if found_separator == -1: #Nu am gasit separator, ci am ajuns la capatul din stanga, deci nu exista un al treilea termen cf regulilor
						sentence_chunk = words[0:second_index]
						attribute_chunk = attributes[0]
						remaining_sentence = " ".join(words[words.index(words[second_index]):len(words)-1])
						attributes.remove(attribute_chunk)
						index = updateIndex(sentence_chunk, index, 1)
						L1, final_index = perform_split_on_targets([remaining_sentence, attributes], index)
						L1_item = [" ".join(sentence_chunk), attribute_chunk]
						L1.append(L1_item)
					elif found_separator != -1 and len(index)==2: #Am gasit separator, dar suntem la finalul propozitiei
						sentence_chunk = words[0:second_index]
						attribute_chunk = attributes[0]
						remaining_sentence = " ".join(words[words.index(words[second_index]):len(words)-1])
						attributes.remove(attribute_chunk)
						index = updateIndex(sentence_chunk, index, 1)
						L1, final_index = perform_split_on_targets([remaining_sentence, attributes], index)
						L1_item = [" ".join(sentence_chunk), attribute_chunk]
						L1.append(L1_item)
					else: #Am gasit separator si nu suntem la finalul propozitiei
						sentence_chunk = words[found_separator:second_index]
						remaining_sentence_1 = " ".join(words[0:found_separator])
						remaining_sentence_2 = " ".join(words[second_index:len(words)-1])
						remaining_sentence = remaining_sentence_1 +' '+ remaining_sentence_2
						attribute_chunk = attributes[len(attributes)-2]
						attributes.remove(attribute_chunk)
						index = updateIndex(sentence_chunk, index, 1)
						L1, final_index = perform_split_on_targets([remaining_sentence, attributes], index)
						L1_item = [" ".join(sentence_chunk), attribute_chunk]
						L1.append(L1_item)
				if found_sep2==1:
					break

	else: #cazul in care 2 termeni sunt nealaturati. Cautam separatorul dintre ei

		index1 = index[0]
		index2 = index[1]
		attributes.sort(key=lambda x: int(x[3]))
		found = -1
		for i in range(index1, index2):
			if words[i] in separator: #am gasit separatorul ce separa propozitiile
				found = 1 
				index_sep = words.index(words[i])
				sentence_chunk = words[0:i]
				attribute_chunk = attributes[0]
				remaining_sentence = " ".join(words[i:])
				index = updateIndex(sentence_chunk, index, 1)
				attributes.remove(attribute_chunk)
				L1, final_index = perform_split_on_targets([remaining_sentence, attributes], index)
				L1_item = [" ".join(sentence_chunk), attribute_chunk]
				L1.append(L1_item)
			if found == 1:
				break
		if found == -1: #Nu am gasit niciun separator, facem split dupa primul termen
			
			sentence_chunk = words[0:index1+1]
			attribute_chunk = attributes[0]
			remaining_sentence = " ".join(words[index1+1:])
			index = updateIndex(sentence_chunk, index, 1)
			attributes.remove(attribute_chunk)
			L1, final_index = perform_split_on_targets([remaining_sentence, attributes], index)
			L1_item = [" ".join(sentence_chunk), attribute_chunk]
			L1.append(L1_item)
			
			
	return L1, final_index
	
def sentence_split(sentence, attributes):
	final_sentences=[]
	final_attributes = []


	target = get_target(attributes) #the target words
	
	if number_of_null_targets(target) >0: #Cazul in care exista targets null
		if number_of_null_targets(target) == len(target): #Cazul in care exista numai targets null
			s, a = perform_split_all_null_targets(sentence, attributes)
			return s, a
		
		else:	#Cazul in care exista si targets nenuli, printre targets null

			index = getIndices(sentence, attributes) #Obtinem indicii la care se gasesc targets
			#Mai intai ne ocupam de propozitiile neadnotate cu target
			returned_list, index = perform_split_remaining_null_targets([sentence, attributes], index) 
			

			for i, item in enumerate(returned_list):
				if i ==0:
					sentences_without_null = item[0]
					attributes_without_null = item[1]
				else:
					final_sentences.append(item[0])
					final_attributes.append(item[1])
					
					
			if final_sentences == []: #Daca, dintr-un motiv ce tine de modul cum a fost scris un review, nu se poate face split pe null terms
				for a in attributes_without_null:
					if a[2]== 'NULL':
						final_sentences.append("Nf")
						final_attributes.append(a)

			
			

			sentence = sentences_without_null #Propozitia care a ramas de prelucrat
			attributes = attributes_without_null #Setul de atribute asociat termenilor, ce au ramas de prelucrat


	index = getIndices(sentence, attributes)
	index.sort()
	#print(index)
	L1, ind = perform_split_on_targets([sentence, attributes], index)
	

	for item in L1:
		final_sentences.append(item[0])
		final_attributes.append(item[1])

	return final_sentences, final_attributes
	
def getFrequenceFeatures(data):	
	vectorizer = CountVectorizer(analyzer='word', lowercase=False,)
	features = vectorizer.fit_transform(data) #Unigram features
	return features, vectorizer
	
def getPresenceFeatures(data):
	vectorizer = CountVectorizer(analyzer='word', lowercase=False,)
	features2 = vectorizer.fit_transform(data)#.toarray() #Unigram features
	
	bin = Binarizer()
	presenceFeatures = bin.fit_transform(features2)
	return presenceFeatures, vectorizer
	
def getBigramFeatures(data):
	vectorizer = CountVectorizer(analyzer='word', lowercase=False, ngram_range=(1,2),)
	features = vectorizer.fit_transform(data)
	
	bin = Binarizer()
	bgFt = bin.fit_transform(features)

	
	return bgFt, vectorizer
	
def getTrigramFeatures(data):
	vectorizer = CountVectorizer(analyzer='word', lowercase=False, ngram_range=(1,3),)
	features = vectorizer.fit_transform(data)
	
	bin = Binarizer()
	tgFt = bin.fit_transform(features)
	return tgFt, vectorizer


negation_words = ["no", "not", "none", "nobody", "nothing", "neither", "nowhere", "never", "hardly", "scarcely", "barely", "doesn't", "isn't", "wasn't", "wouldn't", "couldn't", "won't", "can't", "don't"]
def mark_not(sentences):
	changed_sentences = []
	for sentence in sentences:
		words = word_tokenize(sentence)
		ok = 0
		for i in range(len(words)):
			if ok == 1 and words[i] not in negation_words and words[i] not in string.punctuation:
				words[i] = words[i] + "_NOT"
			if ok == 1 and words[i] in string.punctuation:
				ok = 0
			if words[i] in negation_words:
				if ok == 1:
					ok = 0
				else:
					ok = 1
		changed_sentence = " ".join(words)
		changed_sentences.append(changed_sentence)
	return changed_sentences
	

	
train_file = 'data\\resttrain.xml'
test_file = 'data\\resttest.xml.gold'

train_sentences, train_data = getSentences(train_file)
test_sentences, test_data = getSentences(test_file)

#Prelucrarea propozitiilor
train_sentences = mark_not(train_sentences)
test_sentences= mark_not(test_sentences)






#Creating the train data
train_sentences_prelucrate = []
train_sentences_polarity = []

for i in range(len(train_sentences)):
	polarity_set = set()
	for item in train_data[i]:
		polarity_set.add(item[1])
	if len(polarity_set) > 1:
		fs, fa = sentence_split(train_sentences[i], train_data[i])
		'''
		for i in range(len(fs)):
			if fa[i][1] == 'positive' or fa[i][1] == 'negative' or fa[i][1] == 'neutral':
				train_sentences_prelucrate.append(fs[i])
				train_sentences_polarity.append(fa[i][1])
			else:
				print("WARNING")
		'''
		sentence_snippets = []
		sentence_snippets_polarity = []
		for s in fs:
			sentence_snippets.append(s)
		for a in fa:
			if len(a) != 5:
				#print(a)
				sentence_snippets_polarity.append(a[0])
			else:
				sentence_snippets_polarity.append(a)
				
		for j in range(len(sentence_snippets)):
			train_sentences_prelucrate.append(sentence_snippets[j])
			train_sentences_polarity.append(sentence_snippets_polarity[j][1])
		
	elif len(polarity_set) == 1:
		train_sentences_prelucrate.append(train_sentences[i])
		train_sentences_polarity.append(list(polarity_set)[0])


train_labels = []
for item in train_sentences_polarity:
	if item == 'positive':
		train_labels.append(1)
	elif item == 'negative':
		train_labels.append(2)
	elif item == 'neutral':
		train_labels.append(0)

#Sanity Check
#print(len(train_sentences_prelucrate))
#print(len(train_sentences_polarity))		
#print(len(train_labels))



features, vectorizer = getFrequenceFeatures(train_sentences_prelucrate)
#features, vectorizer = getPresenceFeatures(train_sentences_prelucrate)
#features, vectorizer = getBigramFeatures(train_sentences_prelucrate)
#features, vectorizer = getTrigramFeatures(train_sentences_prelucrate)


print(features.shape)
classifier = LogisticRegression(multi_class='multinomial', solver='newton-cg')
model = classifier.fit(features, train_labels)



#Handling the testing

nr_of_aspects = 0
correct_predictions = 0

for i in range(len(test_sentences)):
	polarity_set = set()
	for item in test_data[i]:
		polarity_set.add(item[1])
	
	if len(polarity_set) > 1:
		#print(test_sentences[i])
		#print(test_data[i])
		#print(polarity_set)
		#print("- - -")
		fs, fa = sentence_split(test_sentences[i], test_data[i])
		
		sentence_snippets = []
		sentence_snippets_polarity = []
		for s in fs:
			sentence_snippets.append(s)
		for a in fa:
			if len(a) != 5:
				#print(a)
				sentence_snippets_polarity.append(a[0])
			else:
				sentence_snippets_polarity.append(a)
				
		for j in range(len(sentence_snippets)):
			#print(sentence_snippets_polarity[j][1], sentence_snippets[j])
			test_features = vectorizer.transform([sentence_snippets[j]])
			prediction = model.predict(test_features)
			nr_of_aspects = nr_of_aspects + 1
			if sentence_snippets_polarity[j][1] == 'neutral':
				true_polarity = 0
			elif sentence_snippets_polarity[j][1] == 'positive':
				true_polarity = 1
			elif sentence_snippets_polarity[j][1] == 'negative':
				true_polarity = 2
				
			if true_polarity == prediction[0]:
				correct_predictions = correct_predictions+1
		
	elif len(polarity_set) == 1:
		test_features = vectorizer.transform([test_sentences[i]])
		prediction = model.predict(test_features)
		nr_of_aspects = nr_of_aspects + len(test_data[i])
		if list(polarity_set)[0] == 'neutral':
			true_polarity = 0
		elif list(polarity_set)[0] == 'positive':
			true_polarity = 1
		elif list(polarity_set)[0] == 'negative':
			true_polarity = 2
		if true_polarity == prediction[0]:
			correct_predictions = correct_predictions + len(test_data[i])

		
print(nr_of_aspects)
print(correct_predictions)
print("Accuracy: ", correct_predictions/nr_of_aspects)
print(features.shape)




	





	
