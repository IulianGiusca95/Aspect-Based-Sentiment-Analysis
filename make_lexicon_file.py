import xml.etree.ElementTree as ET
from lxml import etree
import nltk
from nltk.corpus import stopwords
import string
import numpy as np
import os
from nltk.stem import *

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
translator = str.maketrans('', '', string.punctuation) 
stop_words = set(stopwords.words('english'))

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
						item = [opinion.get('category'), opinion.get('polarity')]
						ac.append(item)
				aspect_categories.append(ac)
				
	return s, aspect_categories
	
def getAspectSentences1(sentences, aspects, lex_aspect):
	aspect_sentences = []
	for i in range(len(sentences)):
		e = []
		for aspect in aspects[i]:
			e_and_a = aspect[0].split("#")
			e.append(e_and_a[0])
		
		if lex_aspect in e:
			aspect_sentences.append(sentences[i])
	return aspect_sentences
	
	
def getAspectSentences2(sentences, aspects, lex_aspect):
	aspect_sentences = []
	for i in range(len(sentences)):
		e = []
		for aspect in aspects[i]:
			e_and_a = aspect[0].split("#")
			e.append(e_and_a[1])
		
		if lex_aspect in e:
			aspect_sentences.append(sentences[i])
	return aspect_sentences

			
def getTokens(sentences):
	tokens = []
	occurences = []
	for sentence in sentences:
		s = sentence.lower().translate(translator)
		words = s.split(" ")
		words_without_stopwords = [word for word in words if word not in stop_words]
	
		for w in words_without_stopwords:
			if w in tokens:
				index = tokens.index(w)
				occurences[index] = occurences[index]+1
			else:
				tokens.append(w)
				occurences.append(1)
				
	return tokens, occurences
	
	
def getTokens_stem(sentences):
	tokens = []
	occurences = []
	for sentence in sentences:
		s = sentence.lower().translate(translator)
		words = s.split(" ")
		words_without_stopwords = [stemmer.stem(word) for word in words if word not in stop_words]
	
		for w in words_without_stopwords:
			if w in tokens:
				index = tokens.index(w)
				occurences[index] = occurences[index]+1
			else:
				tokens.append(w)
				occurences.append(1)
				
	return tokens, occurences
	
	
	
def getTokens_lemma(sentences):
	tokens = []
	occurences = []
	for sentence in sentences:
		s = sentence.lower().translate(translator)
		words = s.split(" ")
		words_without_stopwords = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
	
		for w in words_without_stopwords:
			if w in tokens:
				index = tokens.index(w)
				occurences[index] = occurences[index]+1
			else:
				tokens.append(w)
				occurences.append(1)
				
	return tokens, occurences
	
	
def getTokens_bigram(sentences):
	tokens = []
	occurences = []
	for sentence in sentences:
		s = sentence.lower().translate(translator)
		words = s.split(" ")
		len(words)
	
		for i in range(len(words)-1):
			if words[i]+','+words[i+1] in tokens:
				index = tokens.index(words[i]+','+words[i+1])
				occurences[index] = occurences[index]+1
			else:
				tokens.append(words[i]+','+words[i+1])
				occurences.append(1)
		
				
	return tokens, occurences
	

def makeFile(input_file, output_file, aspect, type):
	sentences, aspects = getSentences(input_file)
	if type == 'E':
		aspect_sentences = getAspectSentences1(sentences, aspects, aspect) 
	
	if type == 'A':
		aspect_sentences = getAspectSentences2(sentences, aspects, aspect) 
	
	tokens, occurences = getTokens(sentences)
	lex_tokens, lex_occurences = getTokens(aspect_sentences)
	if os.path.isfile(output_file):
		os.remove(output_file)
		
	g=open(output_file, "a")
	count=0
	for i in range(len(lex_tokens)):
		#print("Token: ",lex_tokens[i])
		index = tokens.index(lex_tokens[i])
		#print("Appearances in aspect sentences: ",lex_occurences[i])
		#print("Appearances in total: ",occurences[index])
		if lex_occurences[i] > 0 and len(lex_tokens[i]) > 0:
			count = count+1
			precision = lex_occurences[i] / occurences[index]
			recall = lex_occurences[i] / len(aspect_sentences)
			f1 = 2 * precision * recall / (precision + recall)
			#print(lex_tokens[i])
			#print(lex_occurences[i])
			#print("Precision ", precision)
			#print("Recall ", recall)
			#print("F1 ", f1)
			#print(" ")
			info = [lex_tokens[i], lex_occurences[i], precision, recall, f1]
			my_str = ' '.join(str(x) for x in info)
			test_split = my_str.split()
			if len(test_split) < 5:
				print("WARNING: ",lex_tokens[i])
			g.write(my_str + "\n")
	
	
	g.close()
	print(aspect, count)
	
	
	
def makeFile2(input_file, output_file, aspect, type):
	sentences, aspects = getSentences(input_file)
	if type == 'E':
		aspect_sentences = getAspectSentences1(sentences, aspects, aspect) 
	
	if type == 'A':
		aspect_sentences = getAspectSentences2(sentences, aspects, aspect) 
		
	tokens, occurences = getTokens_stem(sentences)
	lex_tokens, lex_occurences = getTokens_stem(aspect_sentences)
	
	if os.path.isfile(output_file):
		os.remove(output_file)
		
	g=open(output_file, "a")
	count=0
	for i in range(len(lex_tokens)):
		#print("Token: ",lex_tokens[i])
		index = tokens.index(lex_tokens[i])
		#print("Appearances in aspect sentences: ",lex_occurences[i])
		#print("Appearances in total: ",occurences[index])
		if lex_occurences[i] > 1 and len(lex_tokens[i]) > 0:
			count = count+1
			precision = lex_occurences[i] / occurences[index]
			recall = lex_occurences[i] / len(aspect_sentences)
			f1 = 2 * precision * recall / (precision + recall)
			#print(lex_tokens[i])
			#print(lex_occurences[i])
			#print("Precision ", precision)
			#print("Recall ", recall)
			#print("F1 ", f1)
			#print(" ")
			info = [lex_tokens[i], lex_occurences[i], precision, recall, f1]
			my_str = ' '.join(str(x) for x in info)
			test_split = my_str.split()
			if len(test_split) < 5:
				print("WARNING: ",lex_tokens[i])
			g.write(my_str + "\n")
	
	
	g.close()
	print(aspect, count)
	
	
def makeFile3(input_file, output_file, aspect, type):
	sentences, aspects = getSentences(input_file)
	if type == 'E':
		aspect_sentences = getAspectSentences1(sentences, aspects, aspect) 
	
	if type == 'A':
		aspect_sentences = getAspectSentences2(sentences, aspects, aspect) 
		
	tokens, occurences = getTokens_bigram(sentences)
	lex_tokens, lex_occurences = getTokens_bigram(aspect_sentences)
	
	if os.path.isfile(output_file):
		os.remove(output_file)
		
	g=open(output_file, "a")
	count=0
	for i in range(len(lex_tokens)):
		#print("Token: ",lex_tokens[i])
		index = tokens.index(lex_tokens[i])
		#print("Appearances in aspect sentences: ",lex_occurences[i])
		#print("Appearances in total: ",occurences[index])
		if lex_occurences[i] > 1 and len(lex_tokens[i]) > 0:
			count = count+1
			precision = lex_occurences[i] / occurences[index]
			recall = lex_occurences[i] / len(aspect_sentences)
			f1 = 2 * precision * recall / (precision + recall)
			#print(lex_tokens[i])
			#print(lex_occurences[i])
			#print("Precision ", precision)
			#print("Recall ", recall)
			#print("F1 ", f1)
			#print(" ")
			info = [lex_tokens[i], lex_occurences[i], precision, recall, f1]
			my_str = ' '.join(str(x) for x in info)
			test_split = my_str.split()
			if len(test_split) < 5:
				print("WARNING: ",lex_tokens[i])
			g.write(my_str + "\n")
	
	
	g.close()
	print(aspect, count)
	

def makeFile4(input_file, output_file, aspect, type):
	sentences, aspects = getSentences(input_file)
	if type == 'E':
		aspect_sentences = getAspectSentences1(sentences, aspects, aspect) 
	
	if type == 'A':
		aspect_sentences = getAspectSentences2(sentences, aspects, aspect) 
		
	tokens, occurences = getTokens_lemma(sentences)
	lex_tokens, lex_occurences = getTokens_lemma(aspect_sentences)
	
	g=open(output_file, "a")
	count=0
	for i in range(len(lex_tokens)):
		#print("Token: ",lex_tokens[i])
		index = tokens.index(lex_tokens[i])
		#print("Appearances in aspect sentences: ",lex_occurences[i])
		#print("Appearances in total: ",occurences[index])
		if lex_occurences[i] > 1 and len(lex_tokens[i]) > 0:
			count = count+1
			precision = lex_occurences[i] / occurences[index]
			recall = lex_occurences[i] / len(aspect_sentences)
			f1 = 2 * precision * recall / (precision + recall)
			#print(lex_tokens[i])
			#print(lex_occurences[i])
			#print("Precision ", precision)
			#print("Recall ", recall)
			#print("F1 ", f1)
			#print(" ")
			info = [lex_tokens[i], lex_occurences[i], precision, recall, f1]
			my_str = ' '.join(str(x) for x in info)
			test_split = my_str.split()
			if len(test_split) < 5:
				print("WARNING: ",lex_tokens[i])
			g.write(my_str + "\n")
	
	
	g.close()
	print(aspect, count)
	


#makeFile("data\\laptops\\train.xml", "unigrams_laptop_lexicon.txt", 'LAPTOP', 'E')
#makeFile("data\\laptops\\train.xml", "unigrams_bat_lexicon.txt", 'BATTERY', 'E')
#makeFile("data\\laptops\\train.xml", "unigrams_cpu_lexicon.txt", 'CPU', 'E')
#makeFile("data\\laptops\\train.xml", "unigrams_graphs_lexicon.txt", 'GRAPHICS', 'E')
#makeFile("data\\laptops\\train.xml", "unigrams_hd_lexicon.txt", 'HARD_DISC', 'E')
#makeFile("data\\laptops\\train.xml", "unigrams_os_lexicon.txt", 'OS', 'E')
#makeFile("data\\laptops\\train.xml", "unigrams_supp_lexicon.txt", 'SUPPORT', 'E')
#makeFile("data\\laptops\\train.xml", "unigrams_comp_lexicon.txt", 'COMPANY', 'E')
#makeFile("data\\laptops\\train.xml", "unigrams_display_lexicon.txt", 'DISPLAY', 'E')
#makeFile("data\\laptops\\train.xml", "unigrams_mouse_lexicon.txt", 'MOUSE', 'E')
#makeFile("data\\laptops\\train.xml", "unigrams_sw_lexicon.txt", 'SOFTWARE', 'E')
#makeFile("data\\laptops\\train.xml", "unigrams_keyb_lexicon.txt", 'KEYBOARD', 'E')
#makeFile("data\\laptops\\train.xml", "unigrams_opt_lexicon.txt", 'OPTICAL_DRIVES', 'E')
#makeFile("data\\laptops\\train.xml", "unigrams_warranty_lexicon.txt", 'WARRANTY', 'E')
#makeFile("data\\laptops\\train.xml", "unigrams_mm_lexicon.txt", 'MULTIMEDIA_DEVICES', 'E')
#makeFile("data\\laptops\\train.xml", "unigrams_ports_lexicon.txt", 'PORTS', 'E')
#makeFile("data\\laptops\\train.xml", "unigrams_power_lexicon.txt", 'POWER_SUPPLY', 'E')
#makeFile("data\\laptops\\train.xml", "unigrams_hw_lexicon.txt", 'HARDWARE', 'E')
#makeFile("data\\laptops\\train.xml", "unigrams_ship_lexicon.txt", 'SHIPPING', 'E')
#makeFile("data\\laptops\\train.xml", "unigrams_mem_lexicon.txt", 'MEMORY', 'E')
#makeFile("data\\laptops\\train.xml", "unigrams_mb_lexicon.txt", 'MOTHERBOARD', 'E')
#makeFile("data\\laptops\\train.xml", "unigrams_fans_lexicon.txt", 'FANS_COOLING', 'E')
#makeFile("data\\laptops\\train.xml", "unigrams_gen_lexicon.txt", 'GENERAL', 'A')
#makeFile("data\\laptops\\train.xml", "unigrams_oper_lexicon.txt", 'OPERATION_PERFORMANCE', 'A')
#makeFile("data\\laptops\\train.xml", "unigrams_des_lexicon.txt", 'DESIGN_FEATURES', 'A')
#makeFile("data\\laptops\\train.xml", "unigrams_use_lexicon.txt", 'USABILITY', 'A')
#makeFile("data\\laptops\\train.xml", "unigrams_port_lexicon.txt", 'PORTABILITY', 'A')
#makeFile("data\\laptops\\train.xml", "unigrams_price_lexicon.txt", 'PRICE', 'A')
#makeFile("data\\laptops\\train.xml", "unigrams_qual_lexicon.txt", 'QUALITY', 'A')
#makeFile("data\\laptops\\train.xml", "unigrams_misc_lexicon.txt", 'MISCELLANEOUS', 'A')
#makeFile("data\\laptops\\train.xml", "unigrams_conn_lexicon.txt", 'CONNECTIVITY', 'A')

#makeFile2("data\\laptops\\train.xml", "stemmed_unigrams_laptop_lexicon.txt", 'LAPTOP', 'E')
#makeFile2("data\\laptops\\train.xml", "stemmed_unigrams_bat_lexicon.txt", 'BATTERY', 'E')
#makeFile2("data\\laptops\\train.xml", "stemmed_unigrams_cpu_lexicon.txt", 'CPU', 'E')
#makeFile2("data\\laptops\\train.xml", "stemmed_unigrams_graphs_lexicon.txt", 'GRAPHICS', 'E')
#makeFile2("data\\laptops\\train.xml", "stemmed_unigrams_hd_lexicon.txt", 'HARD_DISC', 'E')
#makeFile2("data\\laptops\\train.xml", "stemmed_unigrams_os_lexicon.txt", 'OS', 'E')
#makeFile2("data\\laptops\\train.xml", "stemmed_unigrams_supp_lexicon.txt", 'SUPPORT', 'E')
#makeFile2("data\\laptops\\train.xml", "stemmed_unigrams_comp_lexicon.txt", 'COMPANY', 'E')
#makeFile2("data\\laptops\\train.xml", "stemmed_unigrams_display_lexicon.txt", 'DISPLAY', 'E')
#makeFile2("data\\laptops\\train.xml", "stemmed_unigrams_mouse_lexicon.txt", 'MOUSE', 'E')
#makeFile2("data\\laptops\\train.xml", "stemmed_unigrams_sw_lexicon.txt", 'SOFTWARE', 'E')
#makeFile2("data\\laptops\\train.xml", "stemmed_unigrams_keyb_lexicon.txt", 'KEYBOARD', 'E')
#makeFile2("data\\laptops\\train.xml", "stemmed_unigrams_opt_lexicon.txt", 'OPTICAL_DRIVES', 'E')
#makeFile2("data\\laptops\\train.xml", "stemmed_unigrams_warranty_lexicon.txt", 'WARRANTY', 'E')
#makeFile2("data\\laptops\\train.xml", "stemmed_unigrams_mm_lexicon.txt", 'MULTIMEDIA_DEVICES', 'E')
#makeFile2("data\\laptops\\train.xml", "stemmed_unigrams_ports_lexicon.txt", 'PORTS', 'E')
#makeFile2("data\\laptops\\train.xml", "stemmed_unigrams_power_lexicon.txt", 'POWER_SUPPLY', 'E')
#makeFile2("data\\laptops\\train.xml", "stemmed_unigrams_hw_lexicon.txt", 'HARDWARE', 'E')
#makeFile2("data\\laptops\\train.xml", "stemmed_unigrams_ship_lexicon.txt", 'SHIPPING', 'E')
#makeFile2("data\\laptops\\train.xml", "stemmed_unigrams_mem_lexicon.txt", 'MEMORY', 'E')
#makeFile2("data\\laptops\\train.xml", "stemmed_unigrams_mb_lexicon.txt", 'MOTHERBOARD', 'E')
#makeFile2("data\\laptops\\train.xml", "stemmed_unigrams_fans_lexicon.txt", 'FANS_COOLING', 'E')
#makeFile2("data\\laptops\\train.xml", "stemmed_unigrams_gen_lexicon.txt", 'GENERAL', 'A')
#makeFile2("data\\laptops\\train.xml", "stemmed_unigrams_oper_lexicon.txt", 'OPERATION_PERFORMANCE', 'A')
#makeFile2("data\\laptops\\train.xml", "stemmed_unigrams_des_lexicon.txt", 'DESIGN_FEATURES', 'A')
#makeFile2("data\\laptops\\train.xml", "stemmed_unigrams_use_lexicon.txt", 'USABILITY', 'A')
#makeFile2("data\\laptops\\train.xml", "stemmed_unigrams_port_lexicon.txt", 'PORTABILITY', 'A')
#makeFile2("data\\laptops\\train.xml", "stemmed_unigrams_price_lexicon.txt", 'PRICE', 'A')
#makeFile2("data\\laptops\\train.xml", "stemmed_unigrams_qual_lexicon.txt", 'QUALITY', 'A')
#makeFile2("data\\laptops\\train.xml", "stemmed_unigrams_misc_lexicon.txt", 'MISCELLANEOUS', 'A')
#makeFile2("data\\laptops\\train.xml", "stemmed_unigrams_conn_lexicon.txt", 'CONNECTIVITY', 'A')


#makeFile3("data\\laptops\\train.xml", "bigram_laptop_lexicon.txt", 'LAPTOP', 'E')
#makeFile3("data\\laptops\\train.xml", "bigram_bat_lexicon.txt", 'BATTERY', 'E')
#makeFile3("data\\laptops\\train.xml", "bigram_cpu_lexicon.txt", 'CPU', 'E')
#makeFile3("data\\laptops\\train.xml", "bigram_graphs_lexicon.txt", 'GRAPHICS', 'E')
#makeFile3("data\\laptops\\train.xml", "bigram_hd_lexicon.txt", 'HARD_DISC', 'E')
#makeFile3("data\\laptops\\train.xml", "bigram_os_lexicon.txt", 'OS', 'E')
#makeFile3("data\\laptops\\train.xml", "bigram_supp_lexicon.txt", 'SUPPORT', 'E')
#makeFile3("data\\laptops\\train.xml", "bigram_comp_lexicon.txt", 'COMPANY', 'E')
#makeFile3("data\\laptops\\train.xml", "bigram_display_lexicon.txt", 'DISPLAY', 'E')
#makeFile3("data\\laptops\\train.xml", "bigram_mouse_lexicon.txt", 'MOUSE', 'E')
#makeFile3("data\\laptops\\train.xml", "bigram_sw_lexicon.txt", 'SOFTWARE', 'E')
#makeFile3("data\\laptops\\train.xml", "bigram_keyb_lexicon.txt", 'KEYBOARD', 'E')
#makeFile3("data\\laptops\\train.xml", "bigram_opt_lexicon.txt", 'OPTICAL_DRIVES', 'E')
#makeFile3("data\\laptops\\train.xml", "bigram_warranty_lexicon.txt", 'WARRANTY', 'E')
#makeFile3("data\\laptops\\train.xml", "bigram_mm_lexicon.txt", 'MULTIMEDIA_DEVICES', 'E')
#makeFile3("data\\laptops\\train.xml", "bigram_ports_lexicon.txt", 'PORTS', 'E')
#makeFile3("data\\laptops\\train.xml", "bigram_power_lexicon.txt", 'POWER_SUPPLY', 'E')
#makeFile3("data\\laptops\\train.xml", "bigram_hw_lexicon.txt", 'HARDWARE', 'E')
#makeFile3("data\\laptops\\train.xml", "bigram_ship_lexicon.txt", 'SHIPPING', 'E')
#makeFile3("data\\laptops\\train.xml", "bigram_mem_lexicon.txt", 'MEMORY', 'E')
#makeFile3("data\\laptops\\train.xml", "bigram_mb_lexicon.txt", 'MOTHERBOARD', 'E')
#makeFile3("data\\laptops\\train.xml", "bigram_fans_lexicon.txt", 'FANS_COOLING', 'E')
#makeFile3("data\\laptops\\train.xml", "bigram_gen_lexicon.txt", 'GENERAL', 'A')
#makeFile3("data\\laptops\\train.xml", "bigram_oper_lexicon.txt", 'OPERATION_PERFORMANCE', 'A')
#makeFile3("data\\laptops\\train.xml", "bigram_des_lexicon.txt", 'DESIGN_FEATURES', 'A')
#makeFile3("data\\laptops\\train.xml", "bigram_use_lexicon.txt", 'USABILITY', 'A')
#makeFile3("data\\laptops\\train.xml", "bigram_port_lexicon.txt", 'PORTABILITY', 'A')
#makeFile3("data\\laptops\\train.xml", "bigram_price_lexicon.txt", 'PRICE', 'A')
#makeFile3("data\\laptops\\train.xml", "bigram_qual_lexicon.txt", 'QUALITY', 'A')
#makeFile3("data\\laptops\\train.xml", "bigram_misc_lexicon.txt", 'MISCELLANEOUS', 'A')
#makeFile3("data\\laptops\\train.xml", "bigram_conn_lexicon.txt", 'CONNECTIVITY', 'A')


#makeFile4("data\\laptops\\train.xml", "lemma_laptop_lexicon.txt", 'LAPTOP', 'E')
#makeFile4("data\\laptops\\train.xml", "lemma_bat_lexicon.txt", 'BATTERY', 'E')
#makeFile4("data\\laptops\\train.xml", "lemma_cpu_lexicon.txt", 'CPU', 'E')
#makeFile4("data\\laptops\\train.xml", "lemma_graphs_lexicon.txt", 'GRAPHICS', 'E')
#makeFile4("data\\laptops\\train.xml", "lemma_hd_lexicon.txt", 'HARD_DISC', 'E')
#makeFile4("data\\laptops\\train.xml", "lemma_os_lexicon.txt", 'OS', 'E')
#makeFile4("data\\laptops\\train.xml", "lemma_supp_lexicon.txt", 'SUPPORT', 'E')
#makeFile4("data\\laptops\\train.xml", "lemma_comp_lexicon.txt", 'COMPANY', 'E')
#makeFile4("data\\laptops\\train.xml", "lemma_display_lexicon.txt", 'DISPLAY', 'E')
#makeFile4("data\\laptops\\train.xml", "lemma_mouse_lexicon.txt", 'MOUSE', 'E')
#makeFile4("data\\laptops\\train.xml", "lemma_sw_lexicon.txt", 'SOFTWARE', 'E')
#makeFile4("data\\laptops\\train.xml", "lemma_keyb_lexicon.txt", 'KEYBOARD', 'E')
#makeFile4("data\\laptops\\train.xml", "lemma_opt_lexicon.txt", 'OPTICAL_DRIVES', 'E')
#makeFile4("data\\laptops\\train.xml", "lemma_warranty_lexicon.txt", 'WARRANTY', 'E')
#makeFile4("data\\laptops\\train.xml", "lemma_mm_lexicon.txt", 'MULTIMEDIA_DEVICES', 'E')
#makeFile4("data\\laptops\\train.xml", "lemma_ports_lexicon.txt", 'PORTS', 'E')
#makeFile4("data\\laptops\\train.xml", "lemma_power_lexicon.txt", 'POWER_SUPPLY', 'E')
#makeFile4("data\\laptops\\train.xml", "lemma_hw_lexicon.txt", 'HARDWARE', 'E')
#makeFile4("data\\laptops\\train.xml", "lemma_ship_lexicon.txt", 'SHIPPING', 'E')
#makeFile4("data\\laptops\\train.xml", "lemma_mem_lexicon.txt", 'MEMORY', 'E')
#makeFile4("data\\laptops\\train.xml", "lemma_mb_lexicon.txt", 'MOTHERBOARD', 'E')
#makeFile4("data\\laptops\\train.xml", "lemma_fans_lexicon.txt", 'FANS_COOLING', 'E')
#makeFile4("data\\laptops\\train.xml", "lemma_gen_lexicon.txt", 'GENERAL', 'A')
#makeFile4("data\\laptops\\train.xml", "lemma_oper_lexicon.txt", 'OPERATION_PERFORMANCE', 'A')
#makeFile4("data\\laptops\\train.xml", "lemma_des_lexicon.txt", 'DESIGN_FEATURES', 'A')
#makeFile4("data\\laptops\\train.xml", "lemma_use_lexicon.txt", 'USABILITY', 'A')
#makeFile4("data\\laptops\\train.xml", "lemma_port_lexicon.txt", 'PORTABILITY', 'A')
#makeFile4("data\\laptops\\train.xml", "lemma_price_lexicon.txt", 'PRICE', 'A')
#makeFile4("data\\laptops\\train.xml", "lemma_qual_lexicon.txt", 'QUALITY', 'A')
#makeFile4("data\\laptops\\train.xml", "lemma_misc_lexicon.txt", 'MISCELLANEOUS', 'A')
#makeFile4("data\\laptops\\train.xml", "lemma_conn_lexicon.txt", 'CONNECTIVITY', 'A')


makeFile("data\\laptops\\train.xml", "1unigrams_laptop_lexicon.txt", 'LAPTOP', 'E')
makeFile("data\\laptops\\train.xml", "1unigrams_bat_lexicon.txt", 'BATTERY', 'E')
makeFile("data\\laptops\\train.xml", "1unigrams_cpu_lexicon.txt", 'CPU', 'E')
makeFile("data\\laptops\\train.xml", "1unigrams_graphs_lexicon.txt", 'GRAPHICS', 'E')
makeFile("data\\laptops\\train.xml", "1unigrams_hd_lexicon.txt", 'HARD_DISC', 'E')
makeFile("data\\laptops\\train.xml", "1unigrams_os_lexicon.txt", 'OS', 'E')
makeFile("data\\laptops\\train.xml", "1unigrams_supp_lexicon.txt", 'SUPPORT', 'E')
makeFile("data\\laptops\\train.xml", "1unigrams_comp_lexicon.txt", 'COMPANY', 'E')
makeFile("data\\laptops\\train.xml", "1unigrams_display_lexicon.txt", 'DISPLAY', 'E')
makeFile("data\\laptops\\train.xml", "1unigrams_mouse_lexicon.txt", 'MOUSE', 'E')
makeFile("data\\laptops\\train.xml", "1unigrams_sw_lexicon.txt", 'SOFTWARE', 'E')
makeFile("data\\laptops\\train.xml", "1unigrams_keyb_lexicon.txt", 'KEYBOARD', 'E')
makeFile("data\\laptops\\train.xml", "1unigrams_opt_lexicon.txt", 'OPTICAL_DRIVES', 'E')
makeFile("data\\laptops\\train.xml", "1unigrams_warranty_lexicon.txt", 'WARRANTY', 'E')
makeFile("data\\laptops\\train.xml", "1unigrams_mm_lexicon.txt", 'MULTIMEDIA_DEVICES', 'E')
makeFile("data\\laptops\\train.xml", "1unigrams_ports_lexicon.txt", 'PORTS', 'E')
makeFile("data\\laptops\\train.xml", "1unigrams_power_lexicon.txt", 'POWER_SUPPLY', 'E')
makeFile("data\\laptops\\train.xml", "1unigrams_hw_lexicon.txt", 'HARDWARE', 'E')
makeFile("data\\laptops\\train.xml", "1unigrams_ship_lexicon.txt", 'SHIPPING', 'E')
makeFile("data\\laptops\\train.xml", "1unigrams_mem_lexicon.txt", 'MEMORY', 'E')
makeFile("data\\laptops\\train.xml", "1unigrams_mb_lexicon.txt", 'MOTHERBOARD', 'E')
makeFile("data\\laptops\\train.xml", "1unigrams_fans_lexicon.txt", 'FANS_COOLING', 'E')
makeFile("data\\laptops\\train.xml", "1unigrams_gen_lexicon.txt", 'GENERAL', 'A')
makeFile("data\\laptops\\train.xml", "1unigrams_oper_lexicon.txt", 'OPERATION_PERFORMANCE', 'A')
makeFile("data\\laptops\\train.xml", "1unigrams_des_lexicon.txt", 'DESIGN_FEATURES', 'A')
makeFile("data\\laptops\\train.xml", "1unigrams_use_lexicon.txt", 'USABILITY', 'A')
makeFile("data\\laptops\\train.xml", "1unigrams_port_lexicon.txt", 'PORTABILITY', 'A')
makeFile("data\\laptops\\train.xml", "1unigrams_price_lexicon.txt", 'PRICE', 'A')
makeFile("data\\laptops\\train.xml", "1unigrams_qual_lexicon.txt", 'QUALITY', 'A')
makeFile("data\\laptops\\train.xml", "1unigrams_misc_lexicon.txt", 'MISCELLANEOUS', 'A')
makeFile("data\\laptops\\train.xml", "1unigrams_conn_lexicon.txt", 'CONNECTIVITY', 'A')

