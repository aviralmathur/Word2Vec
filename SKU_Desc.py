#!/usr/bin/env python
# 
#  Author: Aviral Mathur (using Angela Chapman's script)
#  Date: 13/8/2015
# The file uses a script for a Kaggle competition prepared by Angela Chapman and can be found at https://github.com/wendykan/DeepLearningMovies/blob/master/Word2Vec_AverageVectors.py
#  However the script, I have modified has been prepared for use for 2 datasets of Crowdflower competition. It finds the cosine similarity between each of the queries in the test.csv file and each of the product titles in train.csv. 
# The interesting thing is that it is using features created from Word2Vec model prepared by the script Word2Vec_AverageVectors.py. 
# This script has been prepared using the Kaggle data for Crowdflower competition https://www.kaggle.com/c/crowdflower-search-relevance
#
# *************************************** #

# ****** Read the two training sets and the test set
#
import pandas as pd
import os
from nltk.corpus import stopwords
import nltk.data
import logging
import numpy as np  # Make sure that numpy is imported
import gensim
from KaggleWord2VecUtility import KaggleWord2VecUtility
from sklearn.metrics.pairwise import cosine_similarity

def makeFeatureVec(words, model, num_features):
    # Function to average all of the word vectors in a given
    # paragraph
    #
    # Pre-initialize an empty numpy array (for speed)
    featureVec = np.zeros((num_features,),dtype="float32")
    #
    nwords = 0.
    #
    # Index2word is a list that contains the names of the words in
    # the model's vocabulary. Convert it to a set, for speed
    index2word_set = set(model.index2word)
    #
    # Loop over each word in the sku and, if it is in the model's
    # vocaublary, add its feature vector to the total
    for word in words:
        if word in index2word_set:
            nwords = nwords + 1.
            featureVec = np.add(featureVec,model[word])
    #
    # Divide the result by the number of words to get the average
    featureVec = np.divide(featureVec,nwords)
    return featureVec


def getAvgFeatureVecs(skucollection, model, num_features):
    # Given a set of skucollection (each one a list of words), calculate
    # the average feature vector for each one and return a 2D numpy array
    #
    # Initialize a counter
    counter = 0.
    #
    # Preallocate a 2D numpy array, for speed
    reviewFeatureVecs = np.zeros((len(skucollection),num_features),dtype="float32")
    #
    # Loop through the skucollection
    for sku in skucollection:
       #
       # Print a status message every 1000th sku
       if counter%1000. == 0.:
           print "sku %d of %d" % (counter, len(skucollection))
       #
       # Call the function (defined above) that makes average feature vectors
       reviewFeatureVecs[counter] = makeFeatureVec(sku, model, \
           num_features)
       #
       # Increment the counter
       counter = counter + 1.
    return reviewFeatureVecs


def getCleanTrainReviews(skucollection):
    clean_skucollection = []
    for sku in skucollection["product_title"]:
        clean_skucollection.append( KaggleWord2VecUtility.sku_to_wordlist( sku, remove_stopwords=False ))
    return clean_skucollection

def getCleanTestReviews(skucollection):
    clean_skucollection = []
    for sku in skucollection["query"]:
        clean_skucollection.append( KaggleWord2VecUtility.sku_to_wordlist( sku, remove_stopwords=False ))
    return clean_skucollection


if __name__ == '__main__':

  	train = pd.read_csv( os.path.join(os.path.dirname(__file__), 'data_crowflower', 'train.csv'), header=0, delimiter=",", quoting=6 )
    	test = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data_crowflower', 'test.csv'), header=0, delimiter=",", quoting=6 )

 	print "Read %d labeled train skucollection, %d labeled test skucollection" % (train["product_title"].size, test["query"].size)

 	# Load the punkt tokenizer
	tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')


   		

	# ****** Set parameters and train the word2vec model
    	#
	# Import the built-in logging module and configure it so that Word2Vec
    	# creates nice output messages
    	logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
	
	num_features = 300    # Word vector dimensionality
	model = gensim.models.Word2Vec.load('300features_40minwords_10_SKU')
	print "Creating average feature vecs for training skucollection"

    	trainDataVecs = getAvgFeatureVecs( getCleanTrainReviews(train), model, num_features )
    
    	
    	print "Creating average feature vecs for test skucollection"

    	testDataVecs = getAvgFeatureVecs( getCleanTestReviews(test), model, num_features )


	print "Query ID, SKU ID, Cosine"
	try:
		query=0
	        for i in testDataVecs:
			isku=0
			for j in trainDataVecs: 
				cos=0.0
				try:
					cos=cosine_similarity(i,j)
				except:
					pass
					
				print "%d, %d, %f"  % (query, isku,cos)
				isku=isku+1
			query=query+1
	except:
		print "exception"
	        pass
