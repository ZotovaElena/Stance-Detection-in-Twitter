# -*- coding: utf-8 -*-

import logging
import pandas as pd
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
import numpy as np
from gensim.models import FastText

#Loading pretrained word2vec model. 
model = FastText.load_fasttext_format('C:/Users/elena.zotova/Desktop/Scripts/w2vec_fasttext_spanish/cc.es.300.bin')
w2v_vectors = model.wv.vectors 
w2v_indices = {word: model.wv.vocab[word].index for word in model.wv.vocab} 
num_features = model.vector_size
vocab = model.wv.vocab.keys()

##How similar are the words?
print (model.wv.similarity('pedir', 'cumplir'))
print (model.wv.most_similar(positive=['sol'], negative=[], topn=5))
vec = model["rey"] - model["mujer"]
ret_vals = model.similar_by_vector(vec)
print(vec, ret_vals)

#out of vocabulary word vector
word = 'Puensi'
v = model[word]
print(v)


print("==========================================")

#make columns for vector features
columns_new = []
count = 0
for c in range(300):
    count = count + 1
    c = "dim_" + str(count)
    columns_new.append(c)

#load datasets 
tweets_train = pd.read_csv('spa_train.csv', sep=';', encoding="utf-8")
tweets_train = tweets_train.fillna('')

tweets_test = pd.read_csv('spa_test.csv', sep=';', encoding="utf-8")
tweets_test = tweets_test.fillna('')

#list of wteets
tweets_train_text = list(tweets_train.full_text.values)
tweets_test_text = list(tweets_test.full_text.values)

#converting strings to tokens
text_data_train = []
for line in tweets_train_text: 
    tokens = line.split()
    text_data_train.append(tokens)
    
text_data_test = []
for line in tweets_test_text: 
    tokens = line.split()
    text_data_test.append(tokens)
    
#functions for calculating an average vector per tweet
def featureVecMethod(words, model, num_features):
    # Pre-initialising empty numpy array for speed
    featureVec = np.zeros(num_features,dtype="float32")
    nwords = 0
    #append a vector to each word
    for word in  words:
        try: 
            nwords = nwords + 1
            v = model[word]
            
            #print(v)
            if np.isnan(v).any():
                print(word, v)
        except KeyError:
            continue
        featureVec = featureVec + model[word]
        print(word, featureVec)
#       
    # dividing the result by number of words to get average
    if nwords != 0:
        featureVec = featureVec/nwords
    return featureVec

def getAvgFeatureVecsOOV(tweets, model, num_features):
    counter = 0
    tweetFeatureVecs = np.zeros((len(tweets),num_features),dtype="float32")
#    all_tweets = len(tweets)
    for i, tweet in enumerate(tweets):
#       # Printing a status message every 1000th review
        if counter%1000 == 0:
            print("Review %d of %d"%(counter,len(tweets)))
            
        tweetFeatureVecs[counter] = featureVecMethod(tweet, model, num_features)
        counter = counter+1 
        
    return tweetFeatureVecs
    
#calculating the average vector
X_Tr = getAvgFeatureVecsOOV(text_data_train, model, num_features)
X_Te = getAvgFeatureVecsOOV(text_data_test, model, num_features)
#removing null vectors 
X_train = []
for v in X_Tr:
    if v.any() != 0.0:
        X_train.append(v)
        
X_test = []
for v in X_Te:
    if v.any() != 0.0:
        X_test.append(v)

#filling columns with features
df_vec_train = pd.DataFrame(X_Tr, columns=columns_new)
df_vec_test = pd.DataFrame(X_Te, columns=columns_new)
#concatenating tweets and features
df_col_merged_train = pd.concat([tweets_train, df_vec_train], axis=1)
df_col_merged_test = pd.concat([tweets_test, df_vec_test], axis=1)

df_full_train = df_col_merged_train[df_col_merged_train.dim_1 != 0.0]
df_full_test = df_col_merged_test[df_col_merged_test.dim_1 != 0.0]

print("=============================================")

from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn import preprocessing
from sklearn.metrics import recall_score

from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

#llabels from the training set
labels_train = list(df_full_train.stance.values)
#Encode the labes to integers
le = preprocessing.LabelEncoder()
le.fit(['AGAINST','FAVOR', 'NEUTRAL'])
labels_int_train = le.transform(labels_train)
#labels from test set
labels_test = list(df_full_test.stance.values)
#Encode the labes to integers
le = preprocessing.LabelEncoder()
le.fit(['AGAINST','FAVOR', 'NEUTRAL'])
labels_int_test = le.transform(labels_test)

#x_train, x_test, y_train, y_test = train_test_split(X, labels_int, test_size=0.1, random_state=0)
x_train = X_train
y_train = labels_int_train

x_test = X_test
y_test = labels_int_test

#making a numpy array
x_train = np.array(x_train, dtype=np.float64)
x_test = np.array(x_test, dtype=np.float64)
y_train = np.array(y_train, dtype=np.float64)
y_test = np.array(y_test, dtype=np.float64)

#number of train and test examples
n_train = len(x_train) 
n_test = len(x_test)
#final matrix for training and testing
x_train = x_train[:n_train]
y_train = y_train[:n_train]
x_test = x_test[:n_test]
y_test = y_test[:n_test]

print("===============================================")
print("Grid search")

#gridsearch for SVM model
from sklearn import svm
from sklearn.model_selection import GridSearchCV
#define the grids of parameters
parameters = {'kernel':['rbf'], 'C':[1, 10, 100, 500], 'gamma': [0.1, 0.001, 0.0001, 0.5, 0.75, 1]}
svc = svm.SVC(gamma="scale")
clf = GridSearchCV(svc, parameters, cv=5, n_jobs=4, verbose=True)
clf.fit(x_train, y_train)
#saving the result of grid search
grid_result = pd.DataFrame(clf.cv_results_)
file_name = "FILENAME.csv"
grid_result.to_csv(file_name, encoding='utf-8', index=False)
#select the best parameters 
df_grid_first = grid_result.loc[grid_result['rank_test_score'] == 1] 
C = list(df_C.pdf_grid_firstaram_C.values)
gamma = list(df_grid_first.param_gamma.values)

print("===============================================")
print("Cross Validation")

from sklearn.model_selection import cross_val_predict
clf = SVC(gamma=gamma, C=C)
clf.fit(x_train, y_train)
y_pred = cross_val_predict(clf, x_train, y_train, cv=10)
target_names=["AGAINST", "FAVOR", "NEUTRAL"]
cl_report = classification_report(y_train, y_pred, target_names=target_names)
cm = confusion_matrix(y_train, y_pred)
print(cm)
print("CROSS VALIDATION", cl_report)

#saving classification reports
import io
report_df = pd.read_fwf(io.StringIO(cl_report), sep="\s+")
file_name = "CV_CR_OOV_spa_without_lemmas.csv"
report_df.to_csv(file_name, encoding='utf-8', index=False)
#saving predicted and wrong predicted examples
df_full_train['predicted'] = y_pred
df_full_train['true'] = y_train
df_full_test.to_csv("spa_OOV_lemma_pred_Cross_Val.txt", encoding='utf-8', index=False)
df1 = df_full_train.loc[df_full_train['predicted'] != df_full_train['true']]
wrong_prediction = df1[['full_text','attribute::id', 'stance', 'predicted', 'true']]
wrong_prediction.to_csv("spa_OOV_without_lemma_wrong_pred_Cross_val.txt", encoding='utf-8', index=False)

print("================================================")
print("Training and Testing")

#define the names of classifiers
names = ["Naive Bayes", "Linear SVM", "RBF SVM", 
         "Decision Tree", "Neural Net", "QDA"]
#choose the classifiers
classifiers = [
    GaussianNB(),
    #set the parameters for SVM obtained from grid search
    SVC(kernel="linear", C=0.025),
    SVC(gamma=1, C=10),
    DecisionTreeClassifier(max_depth=5),
    MLPClassifier(alpha=1, verbose=True),
    QuadraticDiscriminantAnalysis()]
#train models 
for name, clf in zip(names, classifiers):
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    f_score_macro = f1_score(y_test, y_pred, average='macro')
    f_score_micro = f1_score(y_test, y_pred, average='micro')
    precision = precision_score(y_test, y_pred, average='macro') 
    recall = recall_score(y_test, y_pred, average='macro')
    cm = confusion_matrix(y_test, y_pred)
    target_names=["AGAINST", "FAVOR", "NEUTRAL"]
    #save the classification reports
    cl_report = classification_report(y_test, y_pred, target_names=target_names)
    report_df = pd.read_fwf(io.StringIO(cl_report), sep="\s+")
    file_name = "test_CR_OOV_spa_lemma_"+str(name)+".csv"
    report_df.to_csv(file_name, encoding='utf-8', index=False)
    #save the predicted data and wrong predictions
    df_full_test['predicted'] = y_pred
    df_full_test['true'] = y_test
    df_full_test.to_csv("spa_OOV_lemma_pred"+str(name)+".txt", encoding='utf-8', index=False)
    df1 = df_full_test.loc[df_full_test['predicted'] != df_full_test['true']]
    wrong_prediction = df1[['text_lemma','attribute::id', 'stance', 'predicted', 'true']]
    wrong_prediction.to_csv("spa_OOV_lemma_wrong_pred_"+str(name)+".txt", encoding='utf-8', index=False)
    print(cm)
    print(name, 'f-score macro:', f_score_macro)
    print(name, 'f-score micro:', f_score_micro)
    print(name, 'precision: ', precision)
    print(name, 'recall: ', recall)
    print(cl_report)
