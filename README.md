# Aspect-Based-Sentiment-Analysis

I have implemented a system that solves the SemEval2016 task 5. In creating this project, i followed the implementation of Github group of users _nlpaueb_, and expanded on.

ABSA has been split in 3 subtasks:
- Aspect category Detection, solved with Support Vector Machine classifiers and Convolutional Neural Networks;
- Opinion Term Extraction, solved with Conditional Random Fields;
- Polarity Detection, solved with the Multilinear Regression Classifier and text pre-processing.

The code has been documented in the Dissertation paper, available in Romanian, upon request.

# Data required for download

The code has been implemented using Python v. 3.7.2. 

Datasets for training/testing are made available at the following link: http://alt.qcri.org/semeval2016/task5/index.php?id=data-and-tools

The Glove Embedding Vectors (6b) are available for download at: https://nlp.stanford.edu/projects/glove/

Word Embeddings obtained from the Amazon Dataset and the lexicon files: https://www.dropbox.com/s/gmxpovl81y0v656/aux_files.zip?dl=0
