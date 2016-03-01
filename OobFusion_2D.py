'''
OobFusion_2D.py

This code uses two different types of feature vectors (word2vec and n-grams) on the training set to train two random forests.
Probabilities of these two models are then fused by weighted averaging to calculate the final predictions.
'''

import nltk, re, os, pickle, time
import numpy as np
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from nltk import PorterStemmer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import StratifiedShuffleSplit
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer

def review_to_wordlist( review, remove_stopwords=False ):
    # Function to convert a document to a sequence of words,
    # optionally removing stop words.  Returns a list of words.
    #
    # 1. Remove HTML
#     review_text = BeautifulSoup(review).get_text()
    #  
    # 2. Remove non-letters
    review_text = re.sub("[^a-zA-Z]"," ", review)
    #
    # 3. Convert words to lower case and split them
    words = review_text.lower().split()
    #
    # 4. Optionally remove stop words (false by default)
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    #
    # 5. Return a list of words
    return(words)

# Define a function to split a review into parsed sentences
def review_to_sentences( review, tokenizer, remove_stopwords=False ):
    # Function to split a review into parsed sentences. Returns a 
    # list of sentences, where each sentence is a list of words
    #
    # 1. Use the NLTK tokenizer to split the paragraph into sentences
    raw_sentences = tokenizer.tokenize(review.strip())
    #
    # 2. Loop over each sentence
    sentences = []
    for raw_sentence in raw_sentences:
        # If a sentence is empty, skip it
        if len(raw_sentence) > 0:
            # Otherwise, call review_to_wordlist to get a list of words
            sentences.append( review_to_wordlist( raw_sentence, \
              remove_stopwords ))
    #
    # Return the list of sentences (each sentence is a list of words,
    # so this returns a list of lists
    return sentences

def trainWord2Vec(dimensionality, context_window, corpus):
    # train word2vec model
    #
    allSentences= pickle.load( open( corpus, "rb" ) )
    # calculate and return model
    model = Word2Vec(allSentences, size=dimensionality, window=context_window, min_count=1, workers=4)
    return model

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
    # Loop over each word in the review and, if it is in the model's
    # vocaublary, add its feature vector to the total
    for word in words:
        if word in index2word_set: 
            nwords = nwords + 1.
            featureVec = np.add(featureVec,model[word])
    # 
    # Divide the result by the number of words to get the average
    featureVec = np.divide(featureVec,nwords)
    return featureVec


def getAvgFeatureVecs(reviews, model, num_features):
    # Given a set of reviews (each one a list of words), calculate 
    # the average feature vector for each one and return a 2D numpy array 
    # 
    # Initialize a counter
    counter = 0.
    # 
    # Preallocate a 2D numpy array, for speed
    reviewFeatureVecs = np.zeros((len(reviews),num_features),dtype="float32")
    # 
    # Loop through the reviews
    for review in reviews:
#         print(review[0])
        #
        # Print a status message every 1000th review
        if counter%1000. == 0.:
            print ("Document %d of %d" % (counter, len(reviews)))
        # 
        # Call the function (defined above) that makes average feature vectors
        reviewFeatureVecs[counter] = makeFeatureVec(review, model, \
           num_features)
        #
        # Increment the counter
        counter = counter + 1.
    return reviewFeatureVecs

def create_docs_and_labels_variables(datasetDir = "SIMMO/"):
    
    # GET DOCUMENTS AND CONVERT THEM TO W2V FORMAT
    allDocuments_w2v=[]
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    print("Loading sentence words for word2vec...")
    #list category folders of simmo dataset
    categories = os.listdir(datasetDir)
    for category in categories:
    #   list files for each category
        files = os.listdir(datasetDir+category)
        for file in files:
                f = open(datasetDir+category+"/"+file, encoding='utf8')
                text=''
                aggregated = list()
                for line in f:
                    text += line
                sentences = review_to_sentences(text,tokenizer,True)
                # convert list of sentences (where each sentence is a list of words) to an unique list of document words
                for sentence in sentences:
                    aggregated.extend(sentence)
                # add document as a whole to the documents' list, we use append to add the whole list (document) and not only the values (words)
                allDocuments_w2v.append(aggregated)
                
    # GET DOCUMENTS AND CONVERT THEM TO NGRAM FORMAT
    allDocuments_ngrams=[]
    print("Loading sentence words for ngrams...")
    for category in categories:
    #   list files for each category
        files = os.listdir(datasetDir+category)
        for file in files:
            f = open(datasetDir+category+"/"+file, encoding='utf8')
            text=''
            aggregated = list()
            for line in f:
                text += line
            text = re.sub("[^a-zA-Z]"," ", text)
            allDocuments_ngrams.append(text)
            
    # count documents for each category
    categoryCount=dict()
    for category in categories:
        files = os.listdir(datasetDir+category)
        categoryCount[category]=len(files)
    
    #create labels Vector
    print("Creating labels...")
    filesum = categoryCount['Economy']
    economy = filesum
    filesum += categoryCount['Health']
    health = filesum
    filesum += categoryCount['Lifestyle']
    lifestyle = filesum
    filesum += categoryCount['Nature']
    nature = filesum
    filesum += categoryCount['Politics']
    politics = filesum
    filesum += categoryCount['Science']
    science = filesum
    labels=[0 for x in range(science)]
    for count in range(science):
        if count<economy:
            labels[count] = "economy"
        elif count<health:
            labels[count] = "health"
        elif count<lifestyle:
            labels[count] = "lifestyle"
        elif count<nature:
            labels[count] = "nature"
        elif count<politics:
            labels[count] = "politics"
        elif count<science:
            labels[count] = "science"
    
    return allDocuments_w2v, allDocuments_ngrams, labels

def mytokenizer(x):
    # Tokenize sentence and return stemmed words
    #
    stemmed_list = list()
    for y in x.split():
        y_s = PorterStemmer().stem_word(y)
        if len(y_s) > 2:
            stemmed_list.append(y_s)
    return stemmed_list

def readstopwords(file):
    # Read list of stopwords from file (one stopword per line)
    #
    stopwords = list()
    fin = open(file,"r")
    for line in fin:
        stopwords.append(PorterStemmer().stem_word(line.strip()))
#         stopwords.append(line.strip())
    return stopwords

def calculate_metrics(conf_matrix):
    # calculate precision, recall and f-score per class
    #
    length = len(conf_matrix)
    sum_rows = list()
    sum_cols = list()
    precisions = list()
    recalls = list()
    f_scores = list()
    
    # calculate row sums
    for row in conf_matrix:
        sum_r = sum(row)
        sum_rows.append(sum_r)
    # calculate column sums, i = column, j = row
    for i in range(length):
        sum_c=0
        for j in range(length):
            sum_c+= conf_matrix[j][i]
        sum_cols.append(sum_c)    
    # calculate precisions, recalls and f_scores
    for i in range(length):
        correct = conf_matrix[i][i]
        precision = correct / sum_cols[i]
        precisions.append(precision)
        recall = correct / sum_rows[i]
        recalls.append(recall)
        f_score = 2 * precision * recall / (precision + recall)
        f_scores.append(f_score)
        
    macro_precision = np.mean(precisions)
    macro_recall = np.mean(recalls)
    macro_f_score = np.mean(f_scores)
    print("Precisions:", precisions, "Marco average:", macro_precision)
    print("Recalls:", recalls, "Marco average:", macro_recall)
    print("F_scores:", f_scores, "Marco average:", macro_f_score)
        
def calculate_weights(conf_w2v,conf_ngrams,freqs):
    # find late fusion model weights by out-of-the-bag accuracy weighting
    #
    length = len(conf_w2v)
    weights_w2v = list()
    for i in range(length):
        accuracy_w2v = conf_w2v[i][i] / freqs[i]
        accuracy_ngrams = conf_ngrams[i][i] / freqs[i]
        weight = accuracy_w2v / (accuracy_w2v + accuracy_ngrams)
        weights_w2v.append(weight)
    return weights_w2v

# start timer
program_start = time.time()

if(len(sys.argv) == 3):
	# Pickle file where a list of sentences variable is stored to train the word2vec model. Each sentence is a list of words, so a list of lists must be provided.
    corpusFile = sys.argv[1]
	# Directory of dataset (relative path example: "SIMMO/")
    datasetDirectory = sys.argv[2]
else:
	print("You must provide exactly two arguments: First is the corpus pickle file and second is the dataset directory.")
	print("Exiting...")
	exit()

# parse documents into two formats: 1) word2vec format, each document is a list of sentences 2) n-gram format, full text per document with numbers removed
print("Loading documents and labels...")
allDocuments_w2v, allDocuments_ngrams, labels = create_docs_and_labels_variables(datasetDirectory)

 
# balanced random split to dataset and labels
print("Generating stratified random split..")
sss = StratifiedShuffleSplit(labels, 1, test_size=0.3)
for train_index, test_index in sss:
    print("TRAIN indices:", train_index, "TEST indices:", test_index)
    X_train_w2v, X_test_w2v = allDocuments_w2v[train_index], allDocuments_w2v[test_index]
    X_train_ngrams, X_test_ngrams = allDocuments_ngrams[train_index], allDocuments_ngrams[test_index]
    y_train, y_test = labels[train_index], labels[test_index]
    print(len(X_train_w2v),len(X_test_w2v),len(X_train_ngrams),len(X_test_ngrams),len(y_train),len(y_test))


######################### WORD2VEC #########################
print()
print("WORD2VEC")

# extract w2v probabilities on training and test set
word2vecModel = trainWord2Vec(200, 12, corpusFile)
print("\nCalculating training set w2v vectors...")
w2v_train = getAvgFeatureVecs(X_train_w2v, word2vecModel, 200)
print("Calculating test set w2v vectors...")
w2v_test = getAvgFeatureVecs(X_test_w2v, word2vecModel, 200)

# Fit a random forest to the training data, using 1000 trees
forest_w2v = RandomForestClassifier( n_estimators = 1000 , oob_score=True , random_state=1)
print ("Fitting a random forest to labeled training data...")
forest_w2v = forest_w2v.fit( w2v_train, y_train )
 
print ("Oob Score:" + str(forest_w2v.oob_score_) ) 

# get out-of-the-bag predictions
oob_probabilities_w2v = forest_w2v.oob_decision_function_

# variable to access index of classes
classes_w2v = forest_w2v.classes_

oob_predictions_w2v = list()
#for each document
for item in oob_probabilities_w2v:
    # get index of max probability
    max_value = max(item)
    index = np.where(item==max_value)[0][0]
    # get predicted label
    predicted = classes_w2v[index]
    oob_predictions_w2v.append(predicted)
    
# extract confusion matrix
confusion_matrix_oob_w2v = confusion_matrix(y_train, oob_predictions_w2v, labels=["economy", "health", "lifestyle", "nature", "politics", "science"])
print("\nOut-of-the-bag confusion matrix:")
print(confusion_matrix_oob_w2v)

#get class frequencies in training set
class_freqs = list()
print("\nClass frequencies on training set")
for cl in classes_w2v:
    print(cl,":",list(y_train).count(cl))
    class_freqs.append(list(y_train).count(cl))
    

# get model probabilities, predictions and confusion matrix on test_set
probabilities_w2v = forest_w2v.predict_proba(w2v_test)
predictions_w2v = forest_w2v.predict(w2v_test)
confusion_matrix_w2v_test = confusion_matrix(y_test, predictions_w2v, labels=["economy", "health", "lifestyle", "nature", "politics", "science"])
print("\nTest set confusion matrix:")
print(confusion_matrix_w2v_test)


######################### N-GRAMS #########################
print()
print("N-GRAMS")  
print()   

# read stopwords
stopwords = readstopwords("files/stopwords.txt")
    
# get N-gram counts (unigrams, bigrams, trigrams and four-grams)

#unigrams
#train
vectorizer1 = CountVectorizer(tokenizer=mytokenizer, stop_words=stopwords , min_df=0.05)
unigrams = vectorizer1.fit_transform(X_train_ngrams)
tf_unigrams = TfidfTransformer(norm='l1', use_idf=False, smooth_idf = False).fit_transform(unigrams)
tf_total = tf_unigrams
print("Unigrams shape:", np.shape(tf_unigrams))

#test
unigrams_test = vectorizer1.transform(X_test_ngrams)
tf_unigrams_test = TfidfTransformer(norm='l1', use_idf=False, smooth_idf = False).fit_transform(unigrams_test)
tf_total_test = tf_unigrams_test
print("Unigrams shape (test set):", np.shape(tf_unigrams_test))


#bigrams
try:
    #train
    vectorizer2 = CountVectorizer(tokenizer=mytokenizer, stop_words=stopwords , ngram_range=(2,2) , min_df=0.02)
    bigrams = vectorizer2.fit_transform(X_train_ngrams)
    tf_bigrams = TfidfTransformer(norm='l1',use_idf=False, smooth_idf = False).fit_transform(bigrams)
    tf_total = hstack([tf_total,tf_bigrams]).toarray()
    print("Bigrams shape:", np.shape(tf_bigrams))
    
    #test
    bigrams_test = vectorizer2.transform(X_test_ngrams)
    tf_bigrams_test = TfidfTransformer(norm='l1',use_idf=False, smooth_idf = False).fit_transform(bigrams_test)
    tf_total_test = hstack([tf_total_test,tf_bigrams_test]).toarray()
    print("Bigrams shape (test set):", np.shape(tf_bigrams_test))
    
except ValueError:
    print("No bigrams are extracted")  
    
#trigrams   
try:
    #train
    vectorizer3 = CountVectorizer(tokenizer=mytokenizer, stop_words=stopwords , ngram_range=(3,3) , min_df=0.02)
    trigrams = vectorizer3.fit_transform(X_train_ngrams)
    tf_trigrams = TfidfTransformer(norm='l1', use_idf=False, smooth_idf = False).fit_transform(trigrams)
    tf_total = hstack([tf_total,tf_trigrams]).toarray()
    print("Trigrams shape:", np.shape(tf_trigrams))
    
    #test
    trigrams_test = vectorizer3.transform(X_test_ngrams)
    tf_trigrams_test = TfidfTransformer(norm='l1', use_idf=False, smooth_idf = False).fit_transform(trigrams_test)
    tf_total_test = hstack([tf_total_test,tf_trigrams_test]).toarray()
    print("Trigrams shape (test set):", np.shape(tf_trigrams_test))
    
except ValueError as v:
    print("No trigrams are extracted")
    
#four-grams    
try:
    #train
    vectorizer4 = CountVectorizer(tokenizer=mytokenizer, stop_words=stopwords , ngram_range=(4,4) , min_df=0.01)
    fourgrams = vectorizer4.fit_transform(X_train_ngrams)
    tf_fourgrams = TfidfTransformer(norm='l1',use_idf=False, smooth_idf = False).fit_transform(fourgrams)
    tf_total = hstack([tf_total,tf_fourgrams]).toarray()
    print("Fourgrams shape:", np.shape(tf_fourgrams))
    
    #test
    fourgrams_test = vectorizer4.transform(X_test_ngrams)
    tf_fourgrams_test = TfidfTransformer(norm='l1',use_idf=False, smooth_idf = False).fit_transform(fourgrams_test)
    tf_total_test = hstack([tf_total_test,tf_fourgrams_test]).toarray()
    print("Fourgrams shape (test set):", np.shape(tf_fourgrams_test))
    
except ValueError:
    print("No fourgrams are extracted")
    
print("Total vectors shape:",np.shape(tf_total))
print("Total vectors shape (test set):",np.shape(tf_total_test))
print()

# Fit a random forest to the training data, using 1000 trees
forest_ngrams = RandomForestClassifier( n_estimators = 1000 ,oob_score=True, random_state=1)
print ("Fitting a random forest to labeled training data...")
forest_ngrams = forest_ngrams.fit( tf_total, y_train )

print("Out-of-the-bag score:",forest_ngrams.oob_score_)

# get out-of-the-bag predictions
oob_probabilities_ngrams = forest_ngrams.oob_decision_function_
classes_ngrams = forest_ngrams.classes_

oob_predictions_ngrams = list()
for item in oob_probabilities_ngrams:
    # get index of max probability
    max_value = max(item)
    index = np.where(item==max_value)[0][0]
    # get predicted label
    predicted = classes_ngrams[index]
    oob_predictions_ngrams.append(predicted)
    
# extract confusion matrix
confusion_matrix_oob_ngrams = confusion_matrix(y_train, oob_predictions_ngrams, labels=["economy", "health", "lifestyle", "nature", "politics", "science"])
print("\nOut-of-the-bag confusion matrix:")
print(confusion_matrix_oob_ngrams)


# calculate test set predictions and confusion matrix
probabilities_ngrams = forest_ngrams.predict_proba(tf_total_test)
predictions_ngrams = forest_ngrams.predict(tf_total_test)
confusion_matrix_ngrams_test = confusion_matrix(y_test, predictions_ngrams, labels=["economy", "health", "lifestyle", "nature", "politics", "science"])
print("\nTest set confusion matrix:")
print(confusion_matrix_ngrams_test)


# AVERAGE PROBABILITIES OF W2V AND N-GRAM MODELS TO GET FINAL PREDICTIONS
print()
print("Fusing probabilities...")
print()

# set averaging weights
w2v_weights = calculate_weights(confusion_matrix_oob_w2v, confusion_matrix_oob_ngrams , class_freqs)
ngram_weights = [1-w for w in w2v_weights]

# get weighted average probabilities
probabilities_final = list()
for idx, w2v_doc in enumerate(probabilities_w2v):
    ngram_doc = probabilities_ngrams[idx]
#     print(ngram_doc)
    final_doc = list()
    for idx2, w2v_val in enumerate(w2v_doc):
        ngram_val = ngram_doc[idx2]
        final_val = w2v_val * w2v_weights[idx2] + ngram_val * ngram_weights[idx2]
#         print(final_val)
        final_doc.append(final_val)
    probabilities_final.append(final_doc)
 
probabilities_final = np.array(probabilities_final)
print("\nFused test set probabilities matrix:")
print(probabilities_final)

# get final fused predictions
classes = ["economy", "health", "lifestyle", "nature", "politics", "science"]
final_predictions = list()
for item in probabilities_final:
    # get index of max probability
    max_value = max(item)
    index = np.where(item==max_value)[0][0]
    # get predicted label
    predicted = classes[index]
    final_predictions.append(predicted)

# output final fused confusion matrix
confusion_matrix_final = confusion_matrix(y_test, final_predictions, labels=["economy", "health", "lifestyle", "nature", "politics", "science"])
print("\nFused test set confusion matrix:")
print(confusion_matrix_final)

# print elapsed time
program_elapsed = time.time() - program_start
print()
print("Elapsed time (seconds):", program_elapsed)
