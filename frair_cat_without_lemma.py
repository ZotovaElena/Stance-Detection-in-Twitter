# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 09:56:12 2019

@author: elena.zotova
"""

from flair.models import TextClassifier

#build our dataset
#use lemmatized text
import pandas as pd
#load the table 
data = pd.read_csv('cat_train.csv', sep=';', encoding="utf-8").sample(frac=1).drop_duplicates()
#remove emplty rows
data_filtered = data[data['full_text'].notnull()]

data_test_full = pd.read_csv('cat_test.csv', sep=';', encoding="utf-8")
data_test_filtered = data_test_full[data_test_full['full_text'].notnull()]

#convert the data to the Flair format
data_train = data_filtered[['stance', 'full_text']].rename(columns={"stance":"label", "full_text":"text"})
data_test = data_test_filtered[['stance', 'full_text']].rename(columns={"stance":"label", "full_text":"text"})
 
data_train['label'] = '__label__' + data_train['label'].astype(str)
data_test['label'] = '__label__' + data_test['label'].astype(str)

#split data to train, development and test
data_train.iloc[0:int(len(data)*0.9)].to_csv('flair_train.csv', sep='\t', index = False, header = False)
data_train.iloc[int(len(data)*0.9):].to_csv('flair_dev.csv', sep='\t', index = False, header = False)
data_test.to_csv('flair_test.csv', sep='\t', index = False, header = False)

#train a model
from flair.data_fetcher import NLPTaskDataFetcher
from flair.embeddings import WordEmbeddings, FlairEmbeddings, DocumentRNNEmbeddings
from flair.trainers import ModelTrainer
from flair.data import TaggedCorpus
from flair.training_utils import EvaluationMetric
from pathlib import Path


data_folder = Path('./')
corpus: TaggedCorpus = NLPTaskDataFetcher.load_classification_corpus(data_folder, 
                                                                     test_file='flair_test.csv', 
                                                                     dev_file='flair_dev.csv', 
                                                                     train_file='flair_train.csv')
print(corpus)
print(len(corpus.train))
word_embeddings = [WordEmbeddings('ca')] #FastText Wiki 
#set hyperparameters selected after grid search
document_embeddings = DocumentRNNEmbeddings(word_embeddings, 
                                            hidden_size=64, 
                                            reproject_words=True, 
                                            reproject_words_dimension=256,
                                            bidirectional=True,
                                            dropout=0.3169,                                       
                                            rnn_layers=1,
                                            rnn_type='RNN_RELU')
classifier = TextClassifier(document_embeddings, 
                            label_dictionary=corpus.make_label_dictionary(), 
                            multi_label=False)
trainer = ModelTrainer(classifier, corpus)
trainer.train('./', learning_rate=0.05, 
              mini_batch_size=32, 
              max_epochs=10, 
			  evaluation_metric=EvaluationMetric.MACRO_F1_SCORE)


#plot training curves
from flair.visual.training_curves import Plotter
plotter = Plotter()
plotter.plot_training_curves('loss.tsv')
plotter.plot_weights('weights.txt')

#make a table with predictions
test_pred = pd.read_csv('test.tsv', sep='\t', encoding="utf-8", names = ['text', 'label', 'prediction', 'confidence'])
test_full = pd.concat([data_test_filtered, test_pred], axis=1, sort=False)
#make a table with wrong predicions
df = test_full.loc[test_full['prediction'] != test_full['label']]
wrong_prediction = df[['attribute::id', 'text', 'label', 'prediction', 'confidence']]
wrong_prediction.to_csv("cat_without_lemma_flair_wrong_pred.csv", encoding='utf-8', index=False)