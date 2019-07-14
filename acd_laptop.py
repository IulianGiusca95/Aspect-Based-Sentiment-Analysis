import xml.etree.ElementTree as ET
from lxml import etree
import nltk
from nltk.corpus import stopwords
import string
from nltk.stem import PorterStemmer
import numpy as np
from sklearn.svm import SVC
import operator
from nltk.stem import *
from sklearn.feature_extraction.text import CountVectorizer

stop_words = set(stopwords.words('english'))
translator = str.maketrans('', '', string.punctuation)
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

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

def removeUnlabeledSentences(sentences, aspects):
	output_sentences = []
	output_aspects = []
	
	for i in range(len(sentences)):
		if aspects[i] != []:
			output_sentences.append(sentences[i])
			output_aspects.append(aspects[i])
	
	return output_sentences, output_aspects
	
def getEntityAndAttributeSets(aspect_categories):
	entity = []
	attribute = []
	for ac in aspect_categories:
		for item in ac:
			e_and_a = item[0].split('#')
			if e_and_a[0] not in entity:
				entity.append(e_and_a[0])
			if e_and_a[1] not in attribute:
				attribute.append(e_and_a[1])
	
	return entity, attribute
	
def manageLexiconFile(file):
	lexData = []
	f = open(file)
	for line in f:
		features = line.split()
		lexData.append(features)
		
	f.close()
	
	return lexData
				
def load_lexicon(path, entitySet, attributeSet):
	
	entity_lexicon = []
	attribute_lexicon = []
	
	for entity in entitySet:
		if entity == "LAPTOP":
			lexiconData = manageLexiconFile(path+"_laptop_lexicon.txt")
			entity_lexicon.append(lexiconData)
	
		if entity == "BATTERY":
			lexiconData = manageLexiconFile(path+"_bat_lexicon.txt")
			entity_lexicon.append(lexiconData)
			
		if entity == "CPU":
			lexiconData = manageLexiconFile(path+"_cpu_lexicon.txt")
			entity_lexicon.append(lexiconData)
			
		if entity == "GRAPHICS":
			lexiconData = manageLexiconFile(path+"_graphs_lexicon.txt")
			entity_lexicon.append(lexiconData)
			
		if entity == "HARD_DISC":
			lexiconData = manageLexiconFile(path+"_hd_lexicon.txt")
			entity_lexicon.append(lexiconData)
			
		if entity == "OS":
			lexiconData = manageLexiconFile(path+"_os_lexicon.txt")
			entity_lexicon.append(lexiconData)
			
		if entity == "SUPPORT":
			lexiconData = manageLexiconFile(path+"_supp_lexicon.txt")
			entity_lexicon.append(lexiconData)
			
		if entity == "COMPANY":
			lexiconData = manageLexiconFile(path+"_comp_lexicon.txt")
			entity_lexicon.append(lexiconData)
			
		if entity == "DISPLAY":
			lexiconData = manageLexiconFile(path+"_display_lexicon.txt")
			entity_lexicon.append(lexiconData)
			
		if entity == "MOUSE":
			lexiconData = manageLexiconFile(path+"_mouse_lexicon.txt")
			entity_lexicon.append(lexiconData)
			
		if entity == "SOFTWARE":
			lexiconData = manageLexiconFile(path+"_sw_lexicon.txt")
			entity_lexicon.append(lexiconData)
			
		if entity == "KEYBOARD":
			lexiconData = manageLexiconFile(path+"_keyb_lexicon.txt")
			entity_lexicon.append(lexiconData)
			
		if entity == "OPTICAL_DRIVES":
			lexiconData = manageLexiconFile(path+"_opt_lexicon.txt")
			entity_lexicon.append(lexiconData)
			
		if entity == "WARRANTY":
			lexiconData = manageLexiconFile(path+"_warranty_lexicon.txt")
			entity_lexicon.append(lexiconData)
			
		if entity == "MULTIMEDIA_DEVICES":
			lexiconData = manageLexiconFile(path+"_mm_lexicon.txt")
			entity_lexicon.append(lexiconData)
			
		if entity == "PORTS":
			lexiconData = manageLexiconFile(path+"_ports_lexicon.txt")
			entity_lexicon.append(lexiconData)
			
		if entity == "POWER_SUPPLY":
			lexiconData = manageLexiconFile(path+"_power_lexicon.txt")
			entity_lexicon.append(lexiconData)
			
		if entity == "HARDWARE":
			lexiconData = manageLexiconFile(path+"_hw_lexicon.txt")
			entity_lexicon.append(lexiconData)
			
		if entity == "SHIPPING":
			lexiconData = manageLexiconFile(path+"_ship_lexicon.txt")
			entity_lexicon.append(lexiconData)
		
		if entity == "MEMORY":
			lexiconData = manageLexiconFile(path+"_mem_lexicon.txt")
			entity_lexicon.append(lexiconData)
		
		if entity == "MOTHERBOARD":
			lexiconData = manageLexiconFile(path+"_mb_lexicon.txt")
			entity_lexicon.append(lexiconData)
		
		if entity == "FANS_COOLING":
			lexiconData = manageLexiconFile(path+"_fans_lexicon.txt")
			entity_lexicon.append(lexiconData)
		
	
	for attribute in attributeSet:
		if attribute == "GENERAL":
			lexiconData = manageLexiconFile(path+"_gen_lexicon.txt")
			attribute_lexicon.append(lexiconData)
			
		if attribute == "OPERATION_PERFORMANCE":
			lexiconData = manageLexiconFile(path+"_oper_lexicon.txt")
			attribute_lexicon.append(lexiconData)
			
		if attribute == "DESIGN_FEATURES":
			lexiconData = manageLexiconFile(path+"_des_lexicon.txt")
			attribute_lexicon.append(lexiconData)
			
		if attribute == "USABILITY":
			lexiconData = manageLexiconFile(path+"_use_lexicon.txt")
			attribute_lexicon.append(lexiconData)
			
		if attribute == "PORTABILITY":
			lexiconData = manageLexiconFile(path+"_port_lexicon.txt")
			attribute_lexicon.append(lexiconData)
			
		if attribute == "PRICE":
			lexiconData = manageLexiconFile(path+"_price_lexicon.txt")
			attribute_lexicon.append(lexiconData)
			
		if attribute == "QUALITY":
			lexiconData = manageLexiconFile(path+"_qual_lexicon.txt")
			attribute_lexicon.append(lexiconData)
			
		if attribute == "MISCELLANEOUS":
			lexiconData = manageLexiconFile(path+"_misc_lexicon.txt")
			attribute_lexicon.append(lexiconData)
			
		if attribute == "CONNECTIVITY":
			lexiconData = manageLexiconFile(path+"_conn_lexicon.txt")
			attribute_lexicon.append(lexiconData)
			
	return [entity_lexicon, attribute_lexicon]
	
def append_to_list(list, items):
	for item in items:
		list.append(item)
	return list
	
def getFeatures(lexData, words):
	
	#Appending Entity and Attribute lexicons in the same list
	lex = []
	for item in lexData:
		for element in item:
			lex.append(element)
			
	list=[]
	
	for lexicon in lex:
		precision = []
		recall = []
		f1 = []
		for entry in lexicon:
			for w in words:
				if w == entry[0]:
					precision.append(entry[2])
					recall.append(entry[3])
					f1.append(entry[4])
					
		#print(precision)
		min_precision = min(np.array(precision).astype(np.float)) if precision else 0
		max_precision = max(np.array(precision).astype(np.float)) if precision else 0
		avg_precision = np.average(np.array(precision).astype(np.float)) if precision else 0
		median_precision = np.median(np.array(precision).astype(np.float)) if precision else 0
		
		min_recall = min(np.array(recall).astype(np.float)) if recall else 0
		max_recall = max(np.array(recall).astype(np.float)) if recall else 0
		avg_recall = np.average(np.array(recall).astype(np.float)) if recall else 0
		median_recall = np.median(np.array(recall).astype(np.float)) if recall else 0
		
		min_f1 = min(np.array(f1).astype(np.float)) if f1 else 0
		max_f1 = max(np.array(f1).astype(np.float)) if f1 else 0
		avg_f1 = np.average(np.array(f1).astype(np.float)) if f1 else 0
		median_f1 = np.median(np.array(f1).astype(np.float)) if f1 else 0
		
		list = append_to_list(list,[max_precision, min_precision, avg_precision, median_precision, max_recall, min_recall, avg_recall, median_recall, max_f1, min_f1, avg_f1, median_f1])
	
	return list
	
def evaluate(aspects, predicted_aspects):
	common_aspects = 0
	relevant_aspects = 0
	retrieved_aspects = 0
	
	for i in range(len(aspects)):
		correct = set()
		for a in aspects[i]:
			correct.add(a[0])
			
		predicted = set()
		for a in predicted_aspects[i]:
			predicted.add(a)
			
		relevant_aspects = relevant_aspects + len(correct)
		retrieved_aspects = retrieved_aspects + len(predicted)
		common_aspects = common_aspects + len([c for c in predicted if c in correct])
	
	print("Relevant aspects: ", relevant_aspects)
	print("Retrieved aspects: ", retrieved_aspects)
	print("Common aspects: ", common_aspects)
	
	precision = common_aspects / retrieved_aspects if retrieved_aspects > 0 else 0
	recall = common_aspects / relevant_aspects
	f1 = 2 * precision * recall / (precision + recall)
	
	print("Precision: ", precision)
	print("Recall: ", recall)
	print("F1 measure: ", f1)
	
	
	
	
# Dataset
train_file = "data\\laptops\\train.xml"
test_file = "data\\laptops\\test.xml"

# Extragem propozitiile si categoriile din fisier
train_sentences, aspect_categories = getSentences(train_file)
test_sentences, test_categories = getSentences(test_file)


# Inlaturam propozitiile care nu sunt adnotate
train_sentences_extract, aspect_categories_extract = removeUnlabeledSentences(train_sentences, aspect_categories)
test_sentences_extract, test_categories_extract = removeUnlabeledSentences(test_sentences, test_categories)

# Obtinem setul de entitati si atribute
entitySet, attributeSet = getEntityAndAttributeSets(aspect_categories_extract)

# Loading lexicon - lista cu 2 elemente, Entity lexicon si Attribute lexicon
#lex_unigram = load_lexicon("lexica/laptops/unigrams", entitySet, attributeSet)
#lex_stemmed_unigram = load_lexicon("lexica/laptops/stemmed_unigrams", entitySet, attributeSet)

lex_unigram = load_lexicon("lexica2/laptops/unigrams", entitySet, attributeSet)
lex_stemmed_unigram = load_lexicon("lexica2/laptops/stemmed_unigrams", entitySet, attributeSet)
lex_bigram = load_lexicon("lexica2/laptops/bigram", entitySet, attributeSet)
lex_lemmatized_unigram = load_lexicon("lexica2/laptops/lemma", entitySet, attributeSet)
lex_all_unigram = load_lexicon("lexica2/laptops/1unigrams", entitySet, attributeSet)

#Feature vectors marking entities and attribute for each sentence
laptop_vector=[]
battery_vector=[]
cpu_vector=[]
graphics_vector=[]
hard_disc_vector=[]
os_vector=[]
support_vector=[]
company_vector=[]
display_vector=[]
mouse_vector=[]
software_vector=[]
keyboard_vector=[]
optical_drives_vector=[]
warranty_vector=[]
multimedia_devices_vector=[]
ports_vector=[]
power_supply_vector=[]
hardware_vector=[]
shipping_vector=[]
memory_vector=[]
motherboard_vector=[]
fans_cooling_vector=[]

general_vector=[]
operation_performance_vector=[]
design_features_vector=[]
usability_vector=[]
portability_vector=[]
price_vector=[]
quality_vector=[]
miscellaneous_vector=[]
connectivity_vector=[]

vectorizer = CountVectorizer()
vectorizer.fit(train_sentences_extract)



train_sentences_values=[]

#Creating train features
categories = set()

print("Creating training features")
for i in range(len(train_sentences_extract)):
	
	#Transform majuscule in minuscule si elimin semnele de punctuatie
	sentence = train_sentences_extract[i].lower().translate(translator)
	words = sentence.split(" ")
	
	#Inlaturam stopwords
	words_without_stopwords = [word for word in words if word not in stop_words]
	
	#Stemming the words
	stemmed_words = [stemmer.stem(word) for word in words_without_stopwords]
	lemmatized_words = [lemmatizer.lemmatize(word) for word in words_without_stopwords]

	bigram_words = []
	for j in range(len(words)-1):
		bigram_words.append(words[j]+','+words[j+1])

	
	#Pentru fiecare lexicon de entitati sau atribute, returneaza un set de 12 valori, precision, recall, f1 measure
	#Lungimea = nr_of_lex *12
	unigram_features = getFeatures(lex_unigram, words)
	stemmed_unigram_features = getFeatures(lex_stemmed_unigram, stemmed_words)
	bigram_features = getFeatures(lex_bigram, bigram_words)
	#lemmatized_unigram_features = getFeatures(lex_lemmatized_unigram, lemmatized_words)
	#print(unigram_features)
	#print(words)
	#print(len(unigram_features), len(words), len(unigram_features)/len(words))
	#break
	train_sentences_values.append(unigram_features + stemmed_unigram_features + bigram_features)


	
	e, a = getEntityAndAttributeSets([aspect_categories_extract[i]]) #general sets ptr a nu avea duplicate
	for item1 in e:
		for item2 in a:
			categories.add(item1+'#'+item2)
	
	#creating the feature vectors
	#Entity
	if "LAPTOP" in e:
		laptop_vector.append(1)
	else:
		laptop_vector.append(0)
		
	if "BATTERY" in e:
		battery_vector.append(1)
	else:
		battery_vector.append(0)
		
	if "CPU" in e:
		cpu_vector.append(1)
	else:
		cpu_vector.append(0)
		
	if "GRAPHICS" in e:
		graphics_vector.append(1)
	else:
		graphics_vector.append(0)	
		
	if "HARD_DISC" in e:
		hard_disc_vector.append(1)
	else:
		hard_disc_vector.append(0)
		
	if "OS" in e:
		os_vector.append(1)
	else:
		os_vector.append(0)
		
	if "SUPPORT" in e:
		support_vector.append(1)
	else:
		support_vector.append(0)
		
	if "COMPANY" in e:
		company_vector.append(1)
	else:
		company_vector.append(0)
		
	if "DISPLAY" in e:
		display_vector.append(1)
	else:
		display_vector.append(0)
		
	if "MOUSE" in e:
		mouse_vector.append(1)
	else:
		mouse_vector.append(0)
		
	if "SOFTWARE" in e:
		software_vector.append(1)
	else:
		software_vector.append(0)
		
	if "KEYBOARD" in e:
		keyboard_vector.append(1)
	else:
		keyboard_vector.append(0)
		
	if "OPTICAL_DRIVES" in e:
		optical_drives_vector.append(1)
	else:
		optical_drives_vector.append(0)
		
	if "WARRANTY" in e:
		warranty_vector.append(1)
	else:
		warranty_vector.append(0)
		
	if "MULTIMEDIA_DEVICES" in e:
		multimedia_devices_vector.append(1)
	else:
		multimedia_devices_vector.append(0)
		
	if "PORTS" in e:
		ports_vector.append(1)
	else:
		ports_vector.append(0)
		
	if "POWER_SUPPLY" in e:
		power_supply_vector.append(1)
	else:
		power_supply_vector.append(0)
		
	if "HARDWARE" in e:
		hardware_vector.append(1)
	else:
		hardware_vector.append(0)
		
	if "SHIPPING" in e:
		shipping_vector.append(1)
	else:
		shipping_vector.append(0)
		
	if "MEMORY" in e:
		memory_vector.append(1)
	else:
		memory_vector.append(0)
		
	if "MOTHERBOARD" in e:
		motherboard_vector.append(1)
	else:
		motherboard_vector.append(0)
		
	if "FANS_COOLING" in e:
		fans_cooling_vector.append(1)
	else:
		fans_cooling_vector.append(0)
		
	#Attribute
	
	if "GENERAL" in a:
		general_vector.append(1)
	else:
		general_vector.append(0)
		
	if "OPERATION_PERFORMANCE" in a:
		operation_performance_vector.append(1)
	else:
		operation_performance_vector.append(0)
		
	if "DESIGN_FEATURES" in a:
		design_features_vector.append(1)
	else:
		design_features_vector.append(0)
		
	if "USABILITY" in a:
		usability_vector.append(1)
	else:
		usability_vector.append(0)
		
	if "PORTABILITY" in a:
		portability_vector.append(1)
	else:
		portability_vector.append(0)
		
	if "PRICE" in a:
		price_vector.append(1)
	else:
		price_vector.append(0)
		
	if "QUALITY" in a:
		quality_vector.append(1)
	else:
		quality_vector.append(0)
		
	if "MISCELLANEOUS" in a:
		miscellaneous_vector.append(1)
	else:
		miscellaneous_vector.append(0)
		
	if "CONNECTIVITY" in a:
		connectivity_vector.append(1)
	else:
		connectivity_vector.append(0)

categories = list(categories)


print("Creating test features")

test_sentences_values=[]
for i in range(len(test_sentences_extract)):
	#Transform majuscule in minuscule si elimin semnele de punctuatie
	sentence = test_sentences_extract[i].lower().translate(translator)
	words = sentence.split(" ")
	
	#Inlaturam stopwords
	words_without_stopwords = [word for word in words if word not in stop_words]
	
	#Stemming the words
	stemmed_words = [stemmer.stem(word) for word in words_without_stopwords]
	lemmatized_words = [lemmatizer.lemmatize(word) for word in words_without_stopwords]
	
	bigram_words = []
	for j in range(len(words)-1):
		bigram_words.append(words[j]+','+words[j+1])
	
	
	#Pentru fiecare lexicon de entitati sau atribute, returneaza un set de 12 valori, precision, recall, f1 measure
	#Lungimea = nr_of_lex *12
	unigram_features = getFeatures(lex_unigram, words)
	stemmed_unigram_features = getFeatures(lex_stemmed_unigram, stemmed_words)
	bigram_features = getFeatures(lex_bigram, bigram_words)
	#lemmatized_unigram_features = getFeatures(lex_lemmatized_unigram, lemmatized_words)



	test_sentences_values.append(unigram_features + stemmed_unigram_features + bigram_features)
	
	



# Classification
print("Predicting aspects")

train_features = np.asarray(train_sentences_values)
test_features = np.asarray(test_sentences_values)


#train_features = train_sentences_values
#test_features = test_sentences_values



#entity classifiers
laptop_classifier = SVC(kernel='rbf', C=5.656854249492381, gamma=0.02209708691207961, probability=True)
battery_classifier = SVC(kernel='sigmoid', C=22.627416997969522, gamma=0.04419417382415922, probability=True)
cpu_classifier = SVC(kernel='sigmoid', C=64, gamma=0.04419417382415922, probability=True)
graphics_classifier= SVC(kernel='sigmoid', C=128.0, gamma=0.03125, probability=True)
hard_disc_classifier= SVC(kernel='rbf', C=64.0, gamma=0.0078125, probability=True)
os_classifier = SVC(kernel='poly', C=0.000244140625, gamma=4.0, probability=True)
support_classifier = SVC(kernel='rbf', C=38.05462768008707, gamma=0.011048543456039806, probability=True)
company_classifier = SVC(kernel='rbf', C=26.908685288118864, gamma=0.02209708691207961, probability=True)
display_classifier = SVC(kernel='rbf', C=2.378414230005442, gamma=0.02209708691207961, probability=True)
mouse_classifier = SVC(kernel='rbf', C=5.656854249492381, gamma=0.0078125, probability=True)
software_classifier = SVC(kernel='sigmoid', C=6.727171322029716, gamma=0.03716272234383503, probability=True)
keyboard_classifier = SVC(kernel='poly', C=0.0011613350732448448, gamma=1.189207115002721, probability=True)
optical_drives_classifier = SVC(kernel='sigmoid', C=1024, gamma=0.052556025953357156, probability=True)
warranty_classifier = SVC(kernel='sigmoid', C=107.63474115247546, gamma=0.07432544468767006, probability=True)
multimedia_devices_classifier = SVC(kernel='sigmoid', C=304.4370214406966, gamma=0.02209708691207961, probability=True)
ports_classifier = SVC(kernel='linear', C=1.189207115002721, probability=True)
power_supply_classifier = SVC(kernel='sigmoid', C=107.63474115247546, gamma=0.018581361171917516, probability=True)
hardware_classifier = SVC(kernel='sigmoid', C=32, gamma=0.03716272234383503, probability=True)
shipping_classifier = SVC(kernel='linear', C=1.0, probability=True)
memory_classifier = SVC(kernel='sigmoid', C=53.81737057623773, gamma=0.03125, probability=True)
motherboard_classifier = SVC(kernel='linear', C=0.5946035575013605, probability=True)
fans_cooling_classifier = SVC(kernel='sigmoid', C=13.454342644059432, gamma=0.03125, probability=True)


#attribute classifiers
general_classifier = SVC(kernel='rbf', C=3.363585661014858, gamma=0.0625, probability=True)
operation_performance_classifier = SVC(kernel='rbf', C=4.0, gamma=0.125, probability=True)
design_features_classifier = SVC(kernel='rbf', C=2, gamma=0.10511205190671431, probability=True)
usability_classifier = SVC(kernel='rbf', C=13.454342644059432, gamma=0.018581361171917516, probability=True)
portability_classifier = SVC(kernel='rbf', C=2.378414230005442, gamma=0.10511205190671431, probability=True)
price_classifier = SVC(kernel='sigmoid', C=6.727171322029716, gamma=0.03716272234383503, probability=True)
quality_classifier = SVC(kernel='sigmoid', C=4.0, gamma=0.026278012976678578, probability=True)
miscellaneous_classifier = SVC(kernel='rbf', C=1.4142135623730951, gamma=0.10511205190671431, probability=True)
connectivity_classifier = SVC(kernel='sigmoid', C=22.627416997969522, gamma=0.052556025953357156, probability=True)


laptop_classifier.fit(train_features, laptop_vector)
battery_classifier.fit(train_features, battery_vector)
cpu_classifier.fit(train_features, cpu_vector)
graphics_classifier.fit(train_features, graphics_vector)
hard_disc_classifier.fit(train_features, hard_disc_vector)
os_classifier.fit(train_features, os_vector)
support_classifier.fit(train_features, support_vector)
company_classifier.fit(train_features, company_vector)
display_classifier.fit(train_features, display_vector)
mouse_classifier.fit(train_features, mouse_vector)
software_classifier.fit(train_features, software_vector)
keyboard_classifier.fit(train_features, keyboard_vector)
optical_drives_classifier.fit(train_features, optical_drives_vector)
warranty_classifier.fit(train_features, warranty_vector)
multimedia_devices_classifier.fit(train_features, multimedia_devices_vector)
ports_classifier.fit(train_features, ports_vector)
power_supply_classifier.fit(train_features, power_supply_vector)
hardware_classifier.fit(train_features, hardware_vector)
shipping_classifier.fit(train_features, shipping_vector)
memory_classifier.fit(train_features, memory_vector)
motherboard_classifier.fit(train_features, motherboard_vector)
fans_cooling_classifier.fit(train_features, fans_cooling_vector)

general_classifier.fit(train_features, general_vector)
operation_performance_classifier.fit(train_features, operation_performance_vector)
design_features_classifier.fit(train_features, design_features_vector)
usability_classifier.fit(train_features, usability_vector)
portability_classifier.fit(train_features, portability_vector)
price_classifier.fit(train_features, price_vector)
quality_classifier.fit(train_features, quality_vector)
miscellaneous_classifier.fit(train_features, miscellaneous_vector)
connectivity_classifier.fit(train_features, connectivity_vector)

predicted_aspects=[]
for i, test_fvector in enumerate(test_features):
	
	aspects = []
	laptop_prediction = laptop_classifier.predict_proba(test_fvector.reshape(1,-1))[0,1]
	battery_prediction = battery_classifier.predict_proba(test_fvector.reshape(1,-1))[0,1]
	cpu_prediction = cpu_classifier.predict_proba(test_fvector.reshape(1,-1))[0,1]
	graphics_prediction = graphics_classifier.predict_proba(test_fvector.reshape(1,-1))[0,1]
	hard_disc_prediction = hard_disc_classifier.predict_proba(test_fvector.reshape(1,-1))[0,1]
	os_prediction = os_classifier.predict_proba(test_fvector.reshape(1,-1))[0,1]
	support_prediction = support_classifier.predict_proba(test_fvector.reshape(1,-1))[0,1]
	company_prediction = company_classifier.predict_proba(test_fvector.reshape(1,-1))[0,1]
	display_prediction = display_classifier.predict_proba(test_fvector.reshape(1,-1))[0,1]
	mouse_prediction = mouse_classifier.predict_proba(test_fvector.reshape(1,-1))[0,1]
	software_prediction = software_classifier.predict_proba(test_fvector.reshape(1,-1))[0,1]
	keyboard_prediction = keyboard_classifier.predict_proba(test_fvector.reshape(1,-1))[0,1]
	optical_drives_prediction = optical_drives_classifier.predict_proba(test_fvector.reshape(1,-1))[0,1]
	warranty_prediction = warranty_classifier.predict_proba(test_fvector.reshape(1,-1))[0,1]
	multimedia_devices_prediction = multimedia_devices_classifier.predict_proba(test_fvector.reshape(1,-1))[0,1]
	ports_prediction = ports_classifier.predict_proba(test_fvector.reshape(1,-1))[0,1]
	power_supply_prediction = power_supply_classifier.predict_proba(test_fvector.reshape(1,-1))[0,1]
	hardware_prediction = hardware_classifier.predict_proba(test_fvector.reshape(1,-1))[0,1]
	shipping_prediction = shipping_classifier.predict_proba(test_fvector.reshape(1,-1))[0,1]
	memory_prediction = memory_classifier.predict_proba(test_fvector.reshape(1,-1))[0,1]
	motherboard_prediction = motherboard_classifier.predict_proba(test_fvector.reshape(1,-1))[0,1]
	fans_cooling_prediction = fans_cooling_classifier.predict_proba(test_fvector.reshape(1,-1))[0,1]
	
	
	general_prediction = general_classifier.predict_proba(test_fvector.reshape(1,-1))[0,1]
	operation_performance_prediction = operation_performance_classifier.predict_proba(test_fvector.reshape(1,-1))[0,1]
	design_features_prediction = design_features_classifier.predict_proba(test_fvector.reshape(1,-1))[0,1]
	usability_prediction = usability_classifier.predict_proba(test_fvector.reshape(1,-1))[0,1]
	portability_prediction = portability_classifier.predict_proba(test_fvector.reshape(1,-1))[0,1]
	price_prediction = price_classifier.predict_proba(test_fvector.reshape(1,-1))[0,1]
	quality_prediction = quality_classifier.predict_proba(test_fvector.reshape(1,-1))[0,1]
	miscellaneous_prediction = miscellaneous_classifier.predict_proba(test_fvector.reshape(1,-1))[0,1]
	connectivity_prediction = connectivity_classifier.predict_proba(test_fvector.reshape(1,-1))[0,1]
	
	
	entity_probability={"laptop":laptop_prediction, "battery":battery_prediction, "cpu":cpu_prediction, "graphics":graphics_prediction, "hard_disc":hard_disc_prediction, "os":os_prediction, "support":support_prediction, "company":company_prediction, "display":display_prediction, "mouse":mouse_prediction, "software":software_prediction, "keyboard":keyboard_prediction, "optical_drives":optical_drives_prediction, "warranty":warranty_prediction, "multimedia_devices":multimedia_devices_prediction, "ports":ports_prediction, "power_supply":power_supply_prediction, "hardware":hardware_prediction, "shipping":shipping_prediction, "memory":memory_prediction, "motherboard":motherboard_prediction, "fans_cooling":fans_cooling_prediction}
	attribute_probability={"general":general_prediction, "operation_performance":operation_performance_prediction,"design_features":design_features_prediction, "usability":usability_prediction, "portability":portability_prediction, "price":price_prediction, "quality":quality_prediction, "miscellaneous":miscellaneous_prediction, "connectivity":connectivity_prediction}
	
	sorted_entities = sorted(entity_probability.items(), key=operator.itemgetter(1),reverse=True)
	sorted_attributes = sorted(attribute_probability.items(), key=operator.itemgetter(1),reverse=True)
	
	
	#Testam pentru cazurile cand avem mai multe aspecte pe propozitie
	for e in sorted_entities:
		for a in sorted_attributes:
			if e[1] > 0.4 and a[1] > 0.4:
				item = e[0].upper()+'#'+a[0].upper()
				if item in categories:
					aspects.append(item)
	
	#Asignam aspectele cu scorul cel mai mare
	if aspects == []:
		for e in sorted_entities:
			for a in sorted_attributes:
				item = e[0].upper()+'#'+a[0].upper()
				if item in categories:
					aspects.append(item)
					break
			if aspects != []:
				break
		
	predicted_aspects.append(aspects)
	
evaluate(test_categories_extract, predicted_aspects)	






