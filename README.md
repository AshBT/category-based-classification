#category-based-classification

Contains the implementation of a category-based classification framework developed for the MULTISENSOR project. The framework relies on the Random Forests (RF) machine learning method and a late fusion strategy that is based on the operational capabilities of RF. The code has been developed and tested in Python, version 3.5.1, 64-bit.

#Description

In this code, a dataset that contains 12,073 news documents is utilized. The news documents are categorized into six classes (one folder per class). The dataset is available at: http://mklab.iti.gr/files/MULTISENSOR_NewsArticlesData_12073.7z . Two types of textual features are extracted, word2vec and N-grams. Word2vec feature vectors are extracted using the gensim external package. Four groups of N-grams are extracted, namely unigrams, bigrams, trigrams and four-grams. Using a random balanced split on the dataset, one RF model is trained for each type of features. Next, the predicted probabilities from each model on the test set are aggregated, so as to calculate the final late fusion model predictions. These probabilities are not equally weighted in the code. Weights are individually calculated for each class based on the OOB error estimate of each RF model. The output of the code consists of the confusion matrix for each model, including the late fusion model. These matrices can be provided as input, in order to extract evaluation metrics, namely precision, recall and F-score (see function ```calculate_metrics()``` in the code).

#Input arguments

 - A pickle file containing a list of lists variable. This variable corresponds to a corpus that must be provided as input, in order to train the word2vec model by means of the ```trainWord2Vec()``` function. The corpus must be already parsed into a list of sentences, where each sentence is a list of words.
 - The path to a dataset folder.
  
# Version
1.0.0