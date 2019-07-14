import xml.etree.ElementTree as ET
from lxml import etree
import nltk
from nltk.corpus import stopwords
import string
import numpy as np
import os
from nltk.stem import *

translator = str.maketrans('', '', string.punctuation) 

def getSentences(file):
	tree = ET.parse(file, etree.XMLParser(recover=True, encoding="utf-8"))
	root = tree.getroot()
	s = []
	aspect_categories=[]
	for review in root.findall('Review'):
		for sentences in review.findall('sentences'):
			for sentence in sentences.findall('sentence'):
				text = sentence.find("text").text
				s.append(text)
				ac = []
				for opinions in sentence.findall('Opinions'):
					for opinion in opinions.findall('Opinion'):
						item = [opinion.get('category'), opinion.get('polarity'), opinion.get('target')]
						ac.append(item)
				aspect_categories.append(ac)
				
	return s, aspect_categories

def makePOSFile(sentences, output_file):
	
	tag_set = []
	tag_count = []
	for sentence in sentences:
		words = nltk.word_tokenize(sentence)
		tags = nltk.pos_tag(words)
		for _, tag in tags:
			if tag in tag_set:
				ind = tag_set.index(tag)
				tag_count[ind] = tag_count[ind] + 1
			else:
				tag_set.append(tag)
				tag_count.append(1)
	
	if os.path.isfile(output_file):
		os.remove(output_file)
	g=open(output_file, "a")
	for i in range(len(tag_set)):
		if tag_count[i] > 1:
			g.write(tag_set[i]  + "\n")
	g.close()
	
def makeTermFile(aspects, output_file):
	targets = set()
	for a in asp_cat:
		for i in range(len(a)):
			if a[i][2] != 'NULL':
				targets.add(a[i][2])
	targetlist = list(targets)
	if os.path.isfile(output_file):
		os.remove(output_file)	
	g=open(output_file, "a")
	for i in range(len(targetlist)):
		g.write(targetlist[i] + "\n")
	g.close()
	
def makeSuffix1(sentences, output_file):
	suf1 = set()
	for sentence in sentences:
		words = nltk.word_tokenize(sentence)
		for word in words:
			if len(word) >= 1:
				suf1.add(word[len(word)-1])
	#print(suf1)
	suflist = list(suf1)
	if os.path.isfile(output_file):
		os.remove(output_file)
	g=open(output_file, "a")
	for i in range(len(suflist)):
		g.write(suflist[i] + "\n")
	g.close()

def makeSuffix2(sentences, output_file):
	suf1 = set()
	for sentence in sentences:
		words = nltk.word_tokenize(sentence)
		for word in words:
			if len(word) >= 2:
				suf1.add(word[len(word)-2]+word[len(word)-1])
	#print(suf1)
	suflist = list(suf1)
	if os.path.isfile(output_file):
		os.remove(output_file)
	g=open(output_file, "a")
	for i in range(len(suflist)):
		g.write(suflist[i] + "\n")
	g.close()	

def makeSuffix3(sentences, output_file):
	suf1 = set()
	for sentence in sentences:
		words = nltk.word_tokenize(sentence)
		for word in words:
			if len(word) >= 3:
				suf1.add(word[len(word)-3]+word[len(word)-2]+word[len(word)-1])
	#print(suf1)
	suflist = list(suf1)
	if os.path.isfile(output_file):
		os.remove(output_file)
	g=open(output_file, "a")
	for i in range(len(suflist)):
		g.write(suflist[i] + "\n")
	g.close()	
	
def makePrefix1(sentences, output_file):
	pref = set()
	for sentence in sentences:
		words = nltk.word_tokenize(sentence)
		for word in words:
			if len(word) >= 1:
				pref.add(word[0])
	print(pref)
	preflist = list(pref)
	if os.path.isfile(output_file):
		os.remove(output_file)
	g=open(output_file, "a")
	for i in range(len(preflist)):
		g.write(preflist[i] + "\n")
	g.close()
	
def makePrefix2(sentences, output_file):
	pref = set()
	for sentence in sentences:
		words = nltk.word_tokenize(sentence)
		for word in words:
			if len(word) >= 2:
				pref.add(word[0]+word[1])
	print(pref)
	preflist = list(pref)
	if os.path.isfile(output_file):
		os.remove(output_file)
	g=open(output_file, "a")
	for i in range(len(preflist)):
		g.write(preflist[i] + "\n")
	g.close()
	
def makePrefix3(sentences, output_file):
	pref = set()
	for sentence in sentences:
		words = nltk.word_tokenize(sentence)
		for word in words:
			if len(word) >= 3:
				pref.add(word[0]+word[1]+word[2])
	print(pref)
	preflist = list(pref)
	if os.path.isfile(output_file):
		os.remove(output_file)
	g=open(output_file, "a")
	for i in range(len(preflist)):
		g.write(preflist[i] + "\n")
	g.close()

	
sentences, asp_cat = getSentences('data\\restaurants\\train.xml')

#makePOSFile(sentences, 'pos_lexicon.txt')
#makeTermFile(asp_cat, 'term_lexicon.txt')
#makeSuffix1(sentences, 'suffix1_lexicon.txt')
#makeSuffix2(sentences, 'suffix2_lexicon.txt')
#makeSuffix3(sentences, 'suffix3_lexicon.txt')
makePrefix1(sentences, 'prefix1_lexicon.txt')
makePrefix2(sentences, 'prefix2_lexicon.txt')
makePrefix3(sentences, 'prefix3_lexicon.txt')