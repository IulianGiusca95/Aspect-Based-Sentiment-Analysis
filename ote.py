import xml.etree.ElementTree as ET, getopt, logging, sys, random, re, copy, os
import numpy as np
from collections import Counter
import operator
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.svm import SVC
import nltk
from nltk.stem import PorterStemmer
from xml.sax.saxutils import escape
from lxml import etree
from pystruct.models import ChainCRF
from pystruct.learners import FrankWolfeSSVM
#from keras import layers
#from keras.layers import Dense, Embedding, ChainCRF, LSTM, Bidirectional, Dropout
import io
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

def getSentences(file):
    tree = ET.parse(file, etree.XMLParser(recover=True, encoding="utf-8"))
    root = tree.getroot()
    s = []
    aspect_terms = []
    for review in root.findall('Review'):
      for sentences in review.findall('sentences'):
        for sentence in sentences.findall('sentence'):
            if (sentence.get('OutOfScope') != "TRUE"):
              text = sentence.find('text').text
              s.append(text)
              aspect_term = []
              for opinions in sentence.findall('Opinions'):
                  for opinion in opinions.findall('Opinion'):
                      item = [opinion.get('target'), opinion.get('from'), opinion.get('to')]
                      aspect_term.append(item)
              aspect_terms.append(aspect_term)
    return s, aspect_terms
  
#Exista propozitii care nu au atasate opinii/termeni. Ele intregesc contextul
#Le concatenam impreuna cu propozitia care exprima o opinie
def add_Context(sentences, terms):

  length = len(terms)
  i=0
  while(i<length):
    unopinionated_sentences = []
    if terms[i] == []:
        unopinionated_sentences.append(sentences[i])
        i = i+1
        while(i<length and terms[i] == []):
            unopinionated_sentences.append(sentences[i])
            i=i+1
          
        context=''
        for s in unopinionated_sentences:
            context = context + " " + s
        #print(context)
        sentences[i] = sentences[i]  + context
        #print(sentences[i])
        #print('- - - - -')
         
    i=i+1
  
  terms_extract = []
  sentences_extract = []
  for i in range(len(terms)):
    if terms[i] != []:
      terms_extract.append(terms[i])
      sentences_extract.append(sentences[i])
        
  return terms_extract, sentences_extract


def removeSentencesWithoutOpinions(sentences, terms):
  
  sentences1 = []
  terms1 = []
  for i, term in enumerate(terms):
    if term != []:
      terms1.append(term)
      sentences1.append(sentences[i])
  
  return terms1, sentences1
            
    

test_file = "test.xml"
train_file = "train.xml"

#Extragem propozitiile, termenii si pozitiile lor din fisier
#Propozitiile marcate cu tag-ul "OutOfScope" nu sunt luate in considerare
train_sentences, train_terms = getSentences(train_file)
train_terms_extract, train_sentences_extract=removeSentencesWithoutOpinions(train_sentences, train_terms)

#train_terms_extract, train_sentences_extract = add_Context(train_sentences, train_terms)

test_sentences, test_terms = getSentences(test_file)
test_terms_extract, test_sentences_extract = removeSentencesWithoutOpinions(test_sentences, test_terms)
print(len(test_terms_extract))
#test_terms_extract, test_sentences_extract = add_Context(test_sentences, test_terms)

def load_lexicon(file):
  lex=[]
  f = open(file, "r")
  for line in f:
      tag = line.split()[0]
      lex.append(tag)
        
  return lex

prefix1_lexicon = load_lexicon("prefix1_lexicon.txt")
prefix2_lexicon = load_lexicon("prefix2_lexicon.txt")
prefix3_lexicon = load_lexicon("prefix3_lexicon.txt")
suffix1_lexicon = load_lexicon("suffix1_lexicon.txt")
suffix2_lexicon = load_lexicon("suffix2_lexicon.txt")
suffix3_lexicon = load_lexicon("suffix3_lexicon.txt")
term_lexicon = load_lexicon("term_lexicon.txt")
pos_lexicon = load_lexicon("pos_lexicon.txt")

def loadEmbeds(file):
  embed_vectors = {}
  f = open(file)
  for line in f:
    vector = []
    fields = line.split()
    name = fields[0]
    for item in fields[1:]:
      vector.append(float(item))
    embed_vectors[name] = np.asarray(vector)
  f.close()  
  return embed_vectors


embeds = loadEmbeds('word_embeds_restaurants_ote.txt')
print('loaded')

def loadGlove(file):
  glove_vectors = {}
  f = open(file, encoding='utf-8')
  for line in f:
    vector = []
    fields = line.split()
    name = fields[0]
    for item in fields[1:]:
      vector.append(float(item))
    glove_vectors[name] = np.asarray(vector)
  f.close()    
  return glove_vectors

glove_embeds = loadGlove('data\\glove.6B.200d.txt')
print('loaded')

def normalize_embeds(emb):
  feature_vect = []
  norm = np.linalg.norm(emb)
  for v in emb:
    feature_vect.append(v/norm if norm > 0 else 0)
    
  return feature_vect

#Obtain the training features



train_features = []
sentence_labels = [] #B, I or O

for i in range(len(train_sentences_extract)):
  words = nltk.word_tokenize(train_sentences_extract[i])
  #print(words)
  
  last_prediction = ''
  
  tagged_words = nltk.pos_tag(words)
  tag_list = []
  for _, tag in tagged_words:
    tag_list.append(tag)
  
  word_features = []
  word_labels = []

  for j, word in enumerate(words):
    features_prefix_1 = []
    features_prefix_2 = []
    features_prefix_3 = []
    features_suffix_1 = []
    features_suffix_2 = []
    features_suffix_3 = []
    features_frequent_terms = []
    features_pos = []
    features_pos_previous = []
    features_pos_second_previous = []
    features_pos_next = []
    features_pos_second_next = []
    features_morphological = []
    
    features_current = []
    features_prev = []
    features_second_prev = []
    features_third_prev = []
    features_next = []
    features_second_next = []
    features_third_next = []
    
    features_glove_current = []
    features_glove_prev = []
    features_glove_second_prev = []
    features_glove_next = []
    features_glove_second_next = []
    
    #GLOVE
    #Embeds of current word
    if word.lower() in glove_embeds:
      for vector in glove_embeds[word.lower()]:
        features_glove_current.append(vector)
    else:
      count = 1
      word_embed_found = False
      while (j-count) >= 0 and word_embed_found is False:
        if words[j-count].lower() in glove_embeds:
          for vector in glove_embeds[words[j-count].lower()]:
            features_glove_current.append(vector)
          word_embed_found = True
        else:
          count = count + 1
      if word_embed_found is False:
        for vector in embeds['$start1']:
          features_glove_current.append(vector)
          
    #Embeds of previous word
    if j-1 >= 0:
      if words[j-1].lower() in glove_embeds:
        for vector in glove_embeds[words[j-1].lower()]:
          features_glove_prev.append(vector)
      else:
        count = 2
        word_embed_found = False
        while(j - count) >=0 and word_embed_found is False:
          if words[j-count].lower() in glove_embeds:
            for vector in glove_embeds[words[j-count].lower()]:
              features_glove_prev.append(vector)
            word_embed_found = True
          else:
            count = count + 1
        if word_embed_found is False:
          for vector in embeds['$start1']:
            features_glove_prev.append(vector)
    else:
      for vector in embeds['$start1']:
        features_glove_prev.append(vector)
        
    #Embeds of second previous word
    if j-2 >= 0:
      if words[j-2].lower() in glove_embeds:
        for vector in glove_embeds[words[j-2].lower()]:
          features_glove_second_prev.append(vector)
      else:
        count = 3
        word_embed_found = False
        while(j - count) >=0 and word_embed_found is False:
          if words[j-count].lower() in glove_embeds:
            for vector in glove_embeds[words[j-count].lower()]:
              features_glove_second_prev.append(vector)
            word_embed_found = True
          else:
            count = count + 1
        if word_embed_found is False:
          for vector in embeds['$start1']:
            features_glove_second_prev.append(vector)
    else:
      for vector in embeds['$start2']:
        features_glove_second_prev.append(vector)
        
    #Embeds of next word
    if j+1 < len(words):
      if words[j+1].lower() in glove_embeds:
        for vector in glove_embeds[words[j+1].lower()]:
          features_glove_next.append(vector)
      else:
        count = 2
        word_embed_found = False
        while(j + count) <len(words) and word_embed_found is False:
          if words[j+count].lower() in glove_embeds:
            for vector in glove_embeds[words[j+count].lower()]:
              features_glove_next.append(vector)
            word_embed_found = True
          else:
            count = count + 1
        if word_embed_found is False:
          for vector in embeds['$end1']:
            features_glove_next.append(vector)
    else:
      for vector in embeds['$end1']:
        features_glove_next.append(vector)
        
        
    #Embeds of second next
    if j+2 < len(words):
      if words[j+2].lower() in glove_embeds:
        for vector in glove_embeds[words[j+2].lower()]:
          features_glove_second_next.append(vector)
      else:
        count = 3
        word_embed_found = False
        while(j + count) <len(words) and word_embed_found is False:
          if words[j+count].lower() in glove_embeds:
            for vector in glove_embeds[words[j+count].lower()]:
              features_glove_second_next.append(vector)
            word_embed_found = True
          else:
            count = count + 1
        if word_embed_found is False:
          for vector in embeds['$end1']:
            features_glove_second_next.append(vector)
    else:
      for vector in embeds['$end2']:
        features_glove_second_next.append(vector)
        
    
    
    
    
    #NORMAL EMBEDS
    #Embeds of current word
    if word.lower() in embeds:
      for vector in embeds[word.lower()]:
        features_current.append(vector)
    else:
      count = 1
      word_embed_found = False
      while (j-count) >= 0 and word_embed_found is False:
        if words[j-count].lower() in embeds:
          for vector in embeds[words[j-count].lower()]:
            features_current.append(vector)
          word_embed_found = True
        else:
          count = count + 1
      if word_embed_found is False:
        for vector in embeds['$start1']:
          features_current.append(vector)
          
    #Embeds of previous word
    if j-1 >= 0:
      if words[j-1].lower() in embeds:
        for vector in embeds[words[j-1].lower()]:
          features_prev.append(vector)
      else:
        count = 2
        word_embed_found = False
        while(j - count) >=0 and word_embed_found is False:
          if words[j-count].lower() in embeds:
            for vector in embeds[words[j-count].lower()]:
              features_prev.append(vector)
            word_embed_found = True
          else:
            count = count + 1
        if word_embed_found is False:
          for vector in embeds['$start1']:
            features_prev.append(vector)
    else:
      for vector in embeds['$start1']:
        features_prev.append(vector)
        
        
      
    #Embeds of second previous word
    if j-2 >= 0:
      if words[j-2].lower() in embeds:
        for vector in embeds[words[j-2].lower()]:
          features_second_prev.append(vector)
      else:
        count = 3
        word_embed_found = False
        while(j - count) >=0 and word_embed_found is False:
          if words[j-count].lower() in embeds:
            for vector in embeds[words[j-count].lower()]:
              features_second_prev.append(vector)
            word_embed_found = True
          else:
            count = count + 1
        if word_embed_found is False:
          for vector in embeds['$start1']:
            features_second_prev.append(vector)
    else:
      for vector in embeds['$start2']:
        features_second_prev.append(vector)
        
    #Embeds of third previous word
    if j-3 >= 0:
      if words[j-3].lower() in embeds:
        for vector in embeds[words[j-3].lower()]:
          features_third_prev.append(vector)
      else:
        count = 4
        word_embed_found = False
        while(j - count) >=0 and word_embed_found is False:
          if words[j-count].lower() in embeds:
            for vector in embeds[words[j-count].lower()]:
              features_third_prev.append(vector)
            word_embed_found = True
          else:
            count = count + 1
        if word_embed_found is False:
          for vector in embeds['$start1']:
            features_third_prev.append(vector)
    else:
      for vector in embeds['$start2']:
        features_third_prev.append(vector)
        
        
    #Embeds of next word
    if j+1 < len(words):
      if words[j+1].lower() in embeds:
        for vector in embeds[words[j+1].lower()]:
          features_next.append(vector)
      else:
        count = 2
        word_embed_found = False
        while(j + count) <len(words) and word_embed_found is False:
          if words[j+count].lower() in embeds:
            for vector in embeds[words[j+count].lower()]:
              features_next.append(vector)
            word_embed_found = True
          else:
            count = count + 1
        if word_embed_found is False:
          for vector in embeds['$end1']:
            features_next.append(vector)
    else:
      for vector in embeds['$end1']:
        features_next.append(vector)   
              
    #Embeds of second next
    if j+2 < len(words):
      if words[j+2].lower() in embeds:
        for vector in embeds[words[j+2].lower()]:
          features_second_next.append(vector)
      else:
        count = 3
        word_embed_found = False
        while(j + count) <len(words) and word_embed_found is False:
          if words[j+count].lower() in embeds:
            for vector in embeds[words[j+count].lower()]:
              features_second_next.append(vector)
            word_embed_found = True
          else:
            count = count + 1
        if word_embed_found is False:
          for vector in embeds['$end1']:
            features_second_next.append(vector)
    else:
      for vector in embeds['$end2']:
        features_second_next.append(vector)
        
        
    #Embeds of third next
    if j+3 < len(words):
      if words[j+3].lower() in embeds:
        for vector in embeds[words[j+3].lower()]:
          features_third_next.append(vector)
      else:
        count = 4
        word_embed_found = False
        while(j + count) <len(words) and word_embed_found is False:
          if words[j+count].lower() in embeds:
            for vector in embeds[words[j+count].lower()]:
              features_third_next.append(vector)
            word_embed_found = True
          else:
            count = count + 1
        if word_embed_found is False:
          for vector in embeds['$end1']:
            features_third_next.append(vector)
    else:
      for vector in embeds['$end2']:
        features_third_next.append(vector)
        
    features_context = features_current + features_prev + features_second_prev + features_third_prev + features_next + features_second_next + features_third_next
    features_glove_context = features_glove_current + features_glove_prev + features_glove_second_prev + features_glove_next + features_glove_second_next
    features_context_normalized = normalize_embeds(features_context)
    features_glove_context_normalized = normalize_embeds(features_glove_context)
    #features_context = features_current + features_prev + features_next
    #features_context = features_current + features_prev + features_next
    
    target_labels =[]
    #Prefix of length 1 features
    for feature in prefix1_lexicon:
      if feature == word[0]:
        #print(feature, word)
        features_prefix_1.append(1)
      else:
        features_prefix_1.append(0)
    
    #Prefix of length 2 features
    for feature in prefix2_lexicon:
      if len(word)>1:
        if feature == word[0]+word[1]:
          #print(feature, word)
          features_prefix_2.append(1)
        else:
          features_prefix_2.append(0)
      else:
        features_prefix_2.append(0)
    
    #Prefix of length 3 features
    for feature in prefix3_lexicon:
      if len(word)>2:
        if feature == word[0]+word[1]+word[2]:
          #print(feature, word)
          features_prefix_3.append(1)
        else:
          features_prefix_3.append(0)
      else:
        features_prefix_3.append(0)
     
    #Suffix of length 1 
    for feature in suffix1_lexicon:
      if feature == word[-1]:
        #print(feature, word)
        features_suffix_1.append(1)
      else:
        features_suffix_1.append(0)
    
    #Suffix of length 2
    for feature in suffix2_lexicon:
      if len(word)>1:
        if feature == word[-2]+word[-1]:
          #print(feature, word)
          features_suffix_2.append(1)
        else:
          features_suffix_2.append(0)
      else:
        features_suffix_2.append(0)
    
    #Suffix of length 3
    for feature in suffix3_lexicon:
      if len(word)>2:
        if feature == word[-3]+word[-2]+word[-1]:
          #print(feature, word)
          features_suffix_3.append(1)
        else:
          features_suffix_3.append(0)
      else:
        features_suffix_3.append(0)
    
    #Frequent terms
    for feature in term_lexicon:
      if feature == word.lower():
        #print(feature, word)
        features_frequent_terms.append(1)
      else:
        features_frequent_terms.append(0)
        
        
    #POS tag
    for feature in pos_lexicon:
      if feature == tag_list[j]:
        #print(feature, tag_list[j])
        features_pos.append(1)
      else:
        features_pos.append(0)
        
      #Pos tag, previous word
      if j-1 >=0:
        if feature == tag_list[j-1]:
        #print(feature, tag_list[j-1])
          features_pos_previous.append(1)
        else:
          features_pos_previous.append(0)
      else:
        features_pos_previous.append(0)
          
      #Pos tag, second previous word
      if j-2 >=0:
        if feature == tag_list[j-2]:
        #print(feature, tag_list[j-2])
          features_pos_second_previous.append(1)
        else:
          features_pos_second_previous.append(0)
      else:
        features_pos_second_previous.append(0)
          
      #Pos tag, next word
      if j+1 <len(words):
        if feature == tag_list[j+1]:
        #print(feature, tag_list[j+1])
          features_pos_next.append(1)
        else:
          features_pos_next.append(0)
      else:
        features_pos_next.append(0)
          
      #Pos tag, second next word
      if j+2 <len(words):
        if feature == tag_list[j+2]:
          #print(feature, tag_list[j+2])
          features_pos_second_next.append(1)
        else:
          features_pos_second_next.append(0)
      else:
        features_pos_second_next.append(0)
          
          
    #Morphological features:
    #Upper first letter
    if word[0].isupper():
      #print(word)
      features_morphological.append(1)
    else:
      features_morphological.append(0)
      
    capital_letters = 0
    lowercase_letters = 0
    
    for letter in word:
      if letter.isupper():
        capital_letters = capital_letters + 1
      if letter.islower():
        lowercase_letters = lowercase_letters + 1
        
    #Contains capitals, but the first letter is lowercase
    if word[0].islower() and capital_letters >0:
      #print(word)
      features_morphological.append(1)
    else:
      features_morphological.append(0)
      
    #All capitals
    if capital_letters == len(word):
      features_morphological.append(1)
    else:
      features_morphological.append(0)
      
    #All lowercase
    if lowercase_letters == len(word):
      features_morphological.append(1)
    else:
      features_morphological.append(0)
      
    #All digits
    if len(re.findall(r"\d", word)) == len(word):
      features_morphological.append(1)
    else:
      features_morphological.append(0)
      
    #All letters
    if len(re.findall(r"[a-zA-Z]", word)) == len(word):
      #print(word)
      features_morphological.append(1)
    else:
      features_morphological.append(0)
      
    #Contains a dot 
    if len(re.findall(r"[.]", word)) > 0:
      #print(word)
      features_morphological.append(1)
    else:
      features_morphological.append(0)
      
    #Contains a dash
    if len(re.findall(r"[-]", word)) > 0:
      #print(word)
      features_morphological.append(1)
    else:
      features_morphological.append(0)
      
    #Contains any other punctuation mark
    if len(re.findall(r'''[][,;"'?():_`]''', word)) > 0:
      #print(word)
      features_morphological.append(1)
    else:
      features_morphological.append(0)
      
    #Labelling with the IOB System
    term_found = False
  
    aspect_terms = train_terms_extract[i]
    aspect_term_set = set()
    for item in aspect_terms:
      aspect_term_set.add(item[0])
      
    
    for item in aspect_term_set:
      aspect_term = item.lower()
      aspect_term_words = aspect_term.split()
      for index, aspect_word in enumerate(aspect_term_words):
        if word.lower() == aspect_word and (term_found is False):
          if index == 0:
            target_labels = [1] # 1 is B
            last_prediction = "1"
            term_found = True
          else:
            if last_prediction == "1" or last_prediction =="2":
              target_labels = [2] #2 is I
              last_prediction = "2"
              term_found = True
            else:
              target_labels = [0]
              last_prediction = "0"
              
    if term_found is False:
      target_labels = [0]
      last_prediction = "0"
      
    #print(target_labels)
    #features = [features_pos + features_pos_previous + features_pos_second_previous + features_pos_next + features_pos_second_next +
    #           features_morphological + features_frequent_terms + features_prefix_1 + features_prefix_2 + features_prefix_3 + features_suffix_1
    #           + features_suffix_2 + features_suffix_3 + features_context]
    #features = [features_glove_context + features_context]
    #features = [features_pos + features_pos_previous + features_pos_second_previous+ features_pos_next + features_pos_second_next + features_context]
    features = [features_frequent_terms + features_prefix_1 + features_prefix_2 + features_prefix_3 + features_suffix_1 + features_suffix_2 + features_suffix_3 + features_context]
    #features = [features_frequent_terms + features_prefix_1 + features_prefix_2 + features_prefix_3 + features_suffix_1 + features_suffix_2 + features_suffix_3 + features_glove_context_normalized]
    #features = [features_pos + features_pos_previous + features_pos_second_previous + features_pos_next + features_pos_second_next + features_prefix_1 + features_prefix_2 + features_prefix_3 + features_suffix_1 + features_suffix_2 + features_suffix_3 + features_context]
    #features = [features_pos + features_pos_previous + features_pos_second_previous+ features_pos_next + features_pos_second_next + features_context]
    word_features.append(features)
    word_labels.append(target_labels)
      
  #print(len(word_features))
  #if len(word_features[0][0]) != 2737:
  #  print(words)

    
    
  train_sentences_array = np.zeros((len(word_features), len(word_features[0][0])))
  
  index1 = 0
  for word_feature in word_features:
    index2 = 0
    for feat in word_feature:
      for f in feat:
        train_sentences_array[index1, index2] = f
        index2 = index2 + 1
    index1 = index1 + 1
        
  #print(train_sentences_array.shape)
  #for index1, sent in enumerate(train_sentences_array):
  #  for index2, word in enumerate(sent):
  #    if word != 0:
  #      print(word)
  train_features.append(train_sentences_array)
  
  sentence_labels_array = np.zeros((len(word_labels)))
  indexk = 0
  for label in word_labels:
    sentence_labels_array[indexk] = label[0]
    indexk = indexk + 1
  sentence_labels.append(sentence_labels_array.astype(np.int64))


#print(train_features)
#print(sentence_labels)


#Obtaining the test features

test_features = []

for i in range(len(test_sentences_extract)):
  words = nltk.word_tokenize(test_sentences_extract[i])
  #print(words)
  
  
  tagged_words = nltk.pos_tag(words)
  tag_list = []
  for _, tag in tagged_words:
    tag_list.append(tag)
    
 
    
  word_features = []
  for j, word in enumerate(words):
    features_prefix_1 = []
    features_prefix_2 = []
    features_prefix_3 = []
    features_suffix_1 = []
    features_suffix_2 = []
    features_suffix_3 = []
    features_frequent_terms = []
    features_pos = []
    features_pos_previous = []
    features_pos_second_previous = []
    features_pos_next = []
    features_pos_second_next = []
    features_morphological = []
    
    features_current = []
    features_prev = []
    features_second_prev = []
    features_third_prev = []
    features_next = []
    features_second_next = []
    features_third_next = []
    
    features_glove_current = []
    features_glove_prev = []
    features_glove_second_prev = []
    features_glove_next = []
    features_glove_second_next = []
    
    #GLOVE
    #Embeds of current word
    if word.lower() in glove_embeds:
      for vector in glove_embeds[word.lower()]:
        features_glove_current.append(vector)
    else:
      count = 1
      word_embed_found = False
      while (j-count) >= 0 and word_embed_found is False:
        if words[j-count].lower() in glove_embeds:
          for vector in glove_embeds[words[j-count].lower()]:
            features_glove_current.append(vector)
          word_embed_found = True
        else:
          count = count + 1
      if word_embed_found is False:
        for vector in embeds['$start1']:
          features_glove_current.append(vector)
          
    #Embeds of previous word
    if j-1 >= 0:
      if words[j-1].lower() in glove_embeds:
        for vector in glove_embeds[words[j-1].lower()]:
          features_glove_prev.append(vector)
      else:
        count = 2
        word_embed_found = False
        while(j - count) >=0 and word_embed_found is False:
          if words[j-count].lower() in glove_embeds:
            for vector in glove_embeds[words[j-count].lower()]:
              features_glove_prev.append(vector)
            word_embed_found = True
          else:
            count = count + 1
        if word_embed_found is False:
          for vector in embeds['$start1']:
            features_glove_prev.append(vector)
    else:
      for vector in embeds['$start1']:
        features_glove_prev.append(vector)
        
    #Embeds of second previous word
    if j-2 >= 0:
      if words[j-2].lower() in glove_embeds:
        for vector in glove_embeds[words[j-2].lower()]:
          features_glove_second_prev.append(vector)
      else:
        count = 3
        word_embed_found = False
        while(j - count) >=0 and word_embed_found is False:
          if words[j-count].lower() in glove_embeds:
            for vector in glove_embeds[words[j-count].lower()]:
              features_glove_second_prev.append(vector)
            word_embed_found = True
          else:
            count = count + 1
        if word_embed_found is False:
          for vector in embeds['$start1']:
            features_glove_second_prev.append(vector)
    else:
      for vector in embeds['$start2']:
        features_glove_second_prev.append(vector)
        
    #Embeds of next word
    if j+1 < len(words):
      if words[j+1].lower() in glove_embeds:
        for vector in glove_embeds[words[j+1].lower()]:
          features_glove_next.append(vector)
      else:
        count = 2
        word_embed_found = False
        while(j + count) <len(words) and word_embed_found is False:
          if words[j+count].lower() in glove_embeds:
            for vector in glove_embeds[words[j+count].lower()]:
              features_glove_next.append(vector)
            word_embed_found = True
          else:
            count = count + 1
        if word_embed_found is False:
          for vector in embeds['$end1']:
            features_glove_next.append(vector)
    else:
      for vector in embeds['$end1']:
        features_glove_next.append(vector)
        
        
    #Embeds of second next
    if j+2 < len(words):
      if words[j+2].lower() in glove_embeds:
        for vector in glove_embeds[words[j+2].lower()]:
          features_glove_second_next.append(vector)
      else:
        count = 3
        word_embed_found = False
        while(j + count) <len(words) and word_embed_found is False:
          if words[j+count].lower() in glove_embeds:
            for vector in glove_embeds[words[j+count].lower()]:
              features_glove_second_next.append(vector)
            word_embed_found = True
          else:
            count = count + 1
        if word_embed_found is False:
          for vector in embeds['$end1']:
            features_glove_second_next.append(vector)
    else:
      for vector in embeds['$end2']:
        features_glove_second_next.append(vector)
    
    
    #Embeds of current word
    if word.lower() in embeds:
      for vector in embeds[word.lower()]:
        features_current.append(vector)
    else:
      count = 1
      word_embed_found = False
      while (j-count) >= 0 and word_embed_found is False:
        if words[j-count].lower() in embeds:
          for vector in embeds[words[j-count].lower()]:
            features_current.append(vector)
          word_embed_found = True
        else:
          count = count + 1
      if word_embed_found is False:
        for vector in embeds['$start1']:
          features_current.append(vector)
          
    #Embeds of previous word
    if j-1 >= 0:
      if words[j-1].lower() in embeds:
        for vector in embeds[words[j-1].lower()]:
          features_prev.append(vector)
      else:
        count = 2
        word_embed_found = False
        while(j - count) >=0 and word_embed_found is False:
          if words[j-count].lower() in embeds:
            for vector in embeds[words[j-count].lower()]:
              features_prev.append(vector)
            word_embed_found = True
          else:
            count = count + 1
        if word_embed_found is False:
          for vector in embeds['$start1']:
            features_prev.append(vector)
    else:
      for vector in embeds['$start1']:
        features_prev.append(vector)
        
        
      
    #Embeds of second previous word
    if j-2 >= 0:
      if words[j-2].lower() in embeds:
        for vector in embeds[words[j-2].lower()]:
          features_second_prev.append(vector)
      else:
        count = 3
        word_embed_found = False
        while(j - count) >=0 and word_embed_found is False:
          if words[j-count].lower() in embeds:
            for vector in embeds[words[j-count].lower()]:
              features_second_prev.append(vector)
            word_embed_found = True
          else:
            count = count + 1
        if word_embed_found is False:
          for vector in embeds['$start1']:
            features_second_prev.append(vector)
    else:
      for vector in embeds['$start2']:
        features_second_prev.append(vector)
        
    #Embeds of third previous word
    if j-3 >= 0:
      if words[j-3].lower() in embeds:
        for vector in embeds[words[j-3].lower()]:
          features_third_prev.append(vector)
      else:
        count = 4
        word_embed_found = False
        while(j - count) >=0 and word_embed_found is False:
          if words[j-count].lower() in embeds:
            for vector in embeds[words[j-count].lower()]:
              features_third_prev.append(vector)
            word_embed_found = True
          else:
            count = count + 1
        if word_embed_found is False:
          for vector in embeds['$start1']:
            features_third_prev.append(vector)
    else:
      for vector in embeds['$start2']:
        features_third_prev.append(vector)
        
        
    #Embeds of next word
    if j+1 < len(words):
      if words[j+1].lower() in embeds:
        for vector in embeds[words[j+1].lower()]:
          features_next.append(vector)
      else:
        count = 2
        word_embed_found = False
        while(j + count) <len(words) and word_embed_found is False:
          if words[j+count].lower() in embeds:
            for vector in embeds[words[j+count].lower()]:
              features_next.append(vector)
            word_embed_found = True
          else:
            count = count + 1
        if word_embed_found is False:
          for vector in embeds['$end1']:
            features_next.append(vector)
    else:
      for vector in embeds['$end1']:
        features_next.append(vector)   
              
    #Embeds of second next
    if j+2 < len(words):
      if words[j+2].lower() in embeds:
        for vector in embeds[words[j+2].lower()]:
          features_second_next.append(vector)
      else:
        count = 3
        word_embed_found = False
        while(j + count) <len(words) and word_embed_found is False:
          if words[j+count].lower() in embeds:
            for vector in embeds[words[j+count].lower()]:
              features_second_next.append(vector)
            word_embed_found = True
          else:
            count = count + 1
        if word_embed_found is False:
          for vector in embeds['$end1']:
            features_second_next.append(vector)
    else:
      for vector in embeds['$end2']:
        features_second_next.append(vector)
        
        
    #Embeds of second next
    if j+3 < len(words):
      if words[j+3].lower() in embeds:
        for vector in embeds[words[j+3].lower()]:
          features_third_next.append(vector)
      else:
        count = 4
        word_embed_found = False
        while(j + count) <len(words) and word_embed_found is False:
          if words[j+count].lower() in embeds:
            for vector in embeds[words[j+count].lower()]:
              features_third_next.append(vector)
            word_embed_found = True
          else:
            count = count + 1
        if word_embed_found is False:
          for vector in embeds['$end1']:
            features_third_next.append(vector)
    else:
      for vector in embeds['$end2']:
        features_third_next.append(vector)
        
    #features_context = features_current + features_prev + features_second_prev +features_third_prev + features_next + features_second_next + features_third_next
    features_context = features_current + features_prev + features_second_prev +features_third_prev + features_next + features_second_next + features_third_next
    features_glove_context = features_glove_current + features_glove_prev + features_glove_second_prev + features_glove_next + features_glove_second_next
    features_context_normalized = normalize_embeds(features_context)
    features_glove_context_normalized = normalize_embeds(features_glove_context)
    #features_context = features_current + features_prev + features_next
    #features_context = features_current + features_prev + features_next
 
    
    #Prefix of length 1 features
    for feature in prefix1_lexicon:
      if feature == word[0]:
        #print(feature, word)
        features_prefix_1.append(1)
      else:
        features_prefix_1.append(0)
    
    #Prefix of length 2 features
    for feature in prefix2_lexicon:
      if len(word)>1:
        if feature == word[0]+word[1]:
          #print(feature, word)
          features_prefix_2.append(1)
        else:
          features_prefix_2.append(0)
      else:
        features_prefix_2.append(0)
    
    #Prefix of length 3 features
    for feature in prefix3_lexicon:
      if len(word)>2:
        if feature == word[0]+word[1]+word[2]:
          #print(feature, word)
          features_prefix_3.append(1)
        else:
          features_prefix_3.append(0)
      else:
        features_prefix_3.append(0)
     
    #Suffix of length 1 
    for feature in suffix1_lexicon:
      if feature == word[-1]:
        #print(feature, word)
        features_suffix_1.append(1)
      else:
        features_suffix_1.append(0)
    
    #Suffix of length 2
    for feature in suffix2_lexicon:
      if len(word)>1:
        if feature == word[-2]+word[-1]:
          #print(feature, word)
          features_suffix_2.append(1)
        else:
          features_suffix_2.append(0)
      else:
        features_suffix_2.append(0)
    
    #Suffix of length 3
    for feature in suffix3_lexicon:
      if len(word)>2:
        if feature == word[-3]+word[-2]+word[-1]:
          #print(feature, word)
          features_suffix_3.append(1)
        else:
          features_suffix_3.append(0)
      else:
        features_suffix_3.append(0)
    
    #Frequent terms
    for feature in term_lexicon:
      if feature == word.lower():
        #print(feature, word)
        features_frequent_terms.append(1)
      else:
        features_frequent_terms.append(0)
        
        
    #POS tag
    for feature in pos_lexicon:
      if feature == tag_list[j]:
        #print(feature, tag_list[j])
        features_pos.append(1)
      else:
        features_pos.append(0)
        
      #Pos tag, previous word
      if j-1 >=0:
        if feature == tag_list[j-1]:
        #print(feature, tag_list[j-1])
          features_pos_previous.append(1)
        else:
          features_pos_previous.append(0)
      else:
        features_pos_previous.append(0)
          
      #Pos tag, second previous word
      if j-2 >=0:
        if feature == tag_list[j-2]:
        #print(feature, tag_list[j-2])
          features_pos_second_previous.append(1)
        else:
          features_pos_second_previous.append(0)
      else:
        features_pos_second_previous.append(0)
          
      #Pos tag, next word
      if j+1 <len(words):
        if feature == tag_list[j+1]:
        #print(feature, tag_list[j+1])
          features_pos_next.append(1)
        else:
          features_pos_next.append(0)
      else:
        features_pos_next.append(0)
          
      #Pos tag, second next word
      if j+2 <len(words):
        if feature == tag_list[j+2]:
          #print(feature, tag_list[j+2])
          features_pos_second_next.append(1)
        else:
          features_pos_second_next.append(0)
      else:
        features_pos_second_next.append(0)
          
          
    #Morphological features:
    #Upper first letter
    if word[0].isupper():
      #print(word)
      features_morphological.append(1)
    else:
      features_morphological.append(0)
      
    capital_letters = 0
    lowercase_letters = 0
    
    for letter in word:
      if letter.isupper():
        capital_letters = capital_letters + 1
      if letter.islower():
        lowercase_letters = lowercase_letters + 1
        
    #Contains capitals, but the first letter is lowercase
    if word[0].islower() and capital_letters >0:
      #print(word)
      features_morphological.append(1)
    else:
      features_morphological.append(0)
      
    #All capitals
    if capital_letters == len(word):
      features_morphological.append(1)
    else:
      features_morphological.append(0)
      
    #All lowercase
    if lowercase_letters == len(word):
      features_morphological.append(1)
    else:
      features_morphological.append(0)
      
    #All digits
    if len(re.findall(r"\d", word)) == len(word):
      features_morphological.append(1)
    else:
      features_morphological.append(0)
      
    #All letters
    if len(re.findall(r"[a-zA-Z]", word)) == len(word):
      #print(word)
      features_morphological.append(1)
    else:
      features_morphological.append(0)
      
    #Contains a dot 
    if len(re.findall(r"[.]", word)) > 0:
      #print(word)
      features_morphological.append(1)
    else:
      features_morphological.append(0)
      
    #Contains a dash
    if len(re.findall(r"[-]", word)) > 0:
      #print(word)
      features_morphological.append(1)
    else:
      features_morphological.append(0)
      
    #Contains any other punctuation mark
    if len(re.findall(r'''[][,;"'?():_`]''', word)) > 0:
      #print(word)
      features_morphological.append(1)
    else:
      features_morphological.append(0)
      
      
    #features = [features_pos + features_pos_previous + features_pos_second_previous + features_pos_next + features_pos_second_next +
    #           features_morphological + features_frequent_terms + features_prefix_1 + features_prefix_2 + features_prefix_3 + features_suffix_1
    #           + features_suffix_2 + features_suffix_3 + features_context]
    #features = [features_context + features_suffix_1 + features_suffix_2 + features_suffix_3 + features_prefix_1 + features_prefix_2 + features_prefix_3 + features_morphological + features_frequent_terms]
    #features = [features_glove_context + features_context]
    #features = [features_pos + features_pos_previous + features_pos_second_previous+ features_pos_next + features_pos_second_next + features_context]
    features = [features_frequent_terms + features_prefix_1 + features_prefix_2 + features_prefix_3 + features_suffix_1 + features_suffix_2 + features_suffix_3 + features_context]
    #features = [features_morphological + features_prefix_1 + features_prefix_2 + features_prefix_3 + features_suffix_1 + features_suffix_2 + features_suffix_3 + features_context]
    #features = [features_pos + features_pos_previous + features_pos_second_previous + features_pos_next + features_pos_second_next + features_prefix_1 + features_prefix_2 + features_prefix_3 + features_suffix_1 + features_suffix_2 + features_suffix_3 + features_context]
    #features = [features_frequent_terms + features_prefix_1 + features_prefix_2 + features_prefix_3 + features_suffix_1 + features_suffix_2 + features_suffix_3 + features_glove_context_normalized]
    #features = [features_pos + features_pos_previous + features_pos_second_previous+ features_pos_next + features_pos_second_next + features_context]
    word_features.append(features)
    
    
  test_sentences_array = np.zeros((len(word_features), len(word_features[0][0])))
  index1 = 0
  for word_feature in word_features:
    index2 = 0
    for feat in word_feature:
      for f in feat:
        test_sentences_array[index1, index2] = f
        index2 = index2 + 1
    index1 = index1 + 1
  
  #print(test_sentences_array.shape)
  test_features.append(test_sentences_array)
  
model = ChainCRF()
ssvm = FrankWolfeSSVM(model=model, C=.1, max_iter=10)
ssvm.fit(train_features, sentence_labels)

#Predicting the data

def find_offset(s, w):
  start = s.find(w)
  if start == -1:
    start = 0
    end = 0
  else:
    end = start + len(w)
  return start, end

def find_term(s, index):
  words = nltk.word_tokenize(s)
  return words[index]


predictions = ssvm.predict(test_features)
#print(predictions)

predicted_aspect_terms = []
for index1, sentence_prediction in enumerate(predictions):
  aspect_terms_current_sentence = []
  predicted_term = ""
  last_prediction = ""
  for index2, word_prediction in enumerate(sentence_prediction):
    if word_prediction == 1:
      if last_prediction == 1 or last_prediction == 2:
        start, end = find_offset(test_sentences_extract[index1].lower(), predicted_term)
        aspect_terms_current_sentence.append([predicted_term, start, end])
      predicted_term = find_term(test_sentences_extract[index1].lower(), index2)
      last_prediction = 1
    
    elif word_prediction == 2:
      if last_prediction == 1 or last_prediction == 2:
        predterm = find_term(test_sentences_extract[index1].lower(), index2)
        if len(predicted_term) > 0:
          predicted_term = predicted_term + " " + predterm
        else:
          predicted_term = predterm
      last_prediction = 2
      
    elif word_prediction == 0:
      if last_prediction == 1 or last_prediction == 2:
        start, end = find_offset(test_sentences_extract[index1].lower(), predicted_term)
        aspect_terms_current_sentence.append([predicted_term, start, end])
      last_prediction = 0
  if aspect_terms_current_sentence == []:
    aspect_terms_current_sentence.append(['NULL', 0, 0])
  predicted_aspect_terms.append(aspect_terms_current_sentence)
  
  
#Evaluating:

relevant = 0
retrieved = 0
common = 0

for i in range(len(test_sentences_extract)):
  correct_terms = test_terms_extract[i]
      
      
  for term in correct_terms:
      term[1]= int(term[1])
      term[2]= int(term[2])
      
  
  predicted_terms = predicted_aspect_terms[i]
  
  relevant = relevant + len(correct_terms)
  retrieved = retrieved + len(predicted_terms)
  common = common + len([item for item in predicted_terms if item in correct_terms])
  
precision = common/retrieved if retrieved > 0 else 0
recall = common/relevant
f1_measure = 2 * precision * recall / (precision + recall)

print("Precision: ", precision)
print("Recall: ", recall)
print("F1-measure: ", f1_measure)
print("Correct: ", common)
print("Retrieved: ", retrieved)
print("Relevant: ", relevant)