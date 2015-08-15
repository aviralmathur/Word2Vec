#!/usr/bin/env python

#  Author: Angela Chapman
#  Modified: Aviral Mathur
#  Date: 8/6/2014
#
# This file has been modified by Aviral to train the data on a given text file containing a number of SKU descriptions for the Kaggle competition of Crowdflower. The model has been trained to prepare Word2Vec models. Once the model is trained using this script, then use SKU_Desc.py file to find cosine similarities from the feature vectors prepared. 
# https://www.kaggle.com/c/crowdflower-search-relevance
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
from gensim.models import Word2Vec
from sklearn.ensemble import RandomForestClassifier

from KaggleWord2VecUtility import KaggleWord2VecUtility


# ****** Define functions to create average word vectors
#

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


def getCleanReviews(skucollection):
    clean_skucollection = []
    for sku in skucollection["product_title"]:
        clean_skucollection.append( KaggleWord2VecUtility.sku_to_wordlist( sku, remove_stopwords=True ))
    return clean_skucollection



if __name__ == '__main__':

    # Read data from files
    train = pd.read_csv( os.path.join(os.path.dirname(__file__), 'data_crowflower', 'train.csv'), header=0, delimiter=",", quoting=6 )
    

    # Verify the number of skucollection that were read (100,000 in total)
    print "Read %d labeled train skucollection " % (train["product_title"].size)



    # Load the punkt tokenizer
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')



    # ****** Split the labeled and unlabeled training sets into clean sentences
    #
    sentences = []  # Initialize an empty list of sentences

    print "Parsing sentences from training set"
    for sku in train["product_title"]:
        sentences += KaggleWord2VecUtility.sku_to_sentences(sku, tokenizer)

    # ****** Set parameters and train the word2vec model
    #
    # Import the built-in logging module and configure it so that Word2Vec
    # creates nice output messages
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
        level=logging.INFO)

    # Set values for various parameters
    num_features = 300    # Word vector dimensionality
    min_word_count = 40   # Minimum word count
    num_workers = 4       # Number of threads to run in parallel
    context = 10          # Context window size
    downsampling = 1e-3   # Downsample setting for frequent words

    # Initialize and train the model (this will take some time)
    print "Training Word2Vec model..."
    model = Word2Vec(sentences, workers=num_workers, \
                size=num_features, min_count = min_word_count, \
                window = context, sample = downsampling, seed=1)

    # If you don't plan to train the model any further, calling
    # init_sims will make the model much more memory-efficient.
    model.init_sims(replace=True)

    # It can be helpful to create a meaningful model name and
    # save the model for later use. You can load it later using Word2Vec.load()
    model_name = "300features_40minwords_10_SKU"
    model.save(model_name)


