# Word2Vec
Author: Aviral Mathur
Email: aviral.mathur@gmail.com
LinkedIn: https://in.linkedin.com/in/aviralmathur
This python script has been written to find cosine similarity between any 2 text documents using word2vec

For details on word2vec, see
https://code.google.com/p/word2vec/

I am using an implementation called gensim to develop this code, see here to install gensim
https://radimrehurek.com/gensim/install.html

For gensim to run you will need to install anaconda and use its python, it can be found here
http://continuum.io/downloads

Now, the scripts have been developed for the dataset available for the Kaggle challenge at 
https://www.kaggle.com/c/crowdflower-search-relevance 

The scripts are a modification of scripts found here
https://github.com/wendykan/DeepLearningMovies 

Now, to run these scripts do the following:- 

1. You need to create a model using Word2Vec for the exhaustive list of products availble. I train word2vec on the "product title" available for the available filee "train.csv". You need to do the following to train Word2Vec

python Word2Vec_AverageVectors.py

This script will create a Word2Vec model - "300features_40minwords_10_SKU"

2. Now, using this model you can create feature vectors for your text documents. The script finds and prints out the cosine similarity for each of the input customer queries in "test.csv" for each of the SKUs in "train.csv". Please note we first create feature vectors for input customer query and product descriptions. This involves using the word2vec model. After this, for the feature vectors we generate the cosine similarity. You need to do the below for printing the cosine similarity

python SKU_Desc.py

This will print the cosine similarities in the below format
customer query, sku id, cosine similarity

