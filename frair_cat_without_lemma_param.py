# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 09:56:12 2019

@author: elena.zotova
"""

#build our dataset
#use lemmatized text
import pandas as pd
data = pd.read_csv('cat_train.csv', sep=';', encoding="utf-8").sample(frac=1).drop_duplicates()
data_test = pd.read_csv('cat_test.csv', sep=';', encoding="utf-8")
data = data[['stance', 'full_text']].rename(columns={"stance":"label", "full_text":"text"})
data_test = data_test[['stance', 'full_text']].rename(columns={"stance":"label", "full_text":"text"})
 
data['label'] = '__label__' + data['label'].astype(str)
data_test['label'] = '__label__' + data['label'].astype(str)
data.iloc[0:int(len(data)*0.9)].to_csv('flair_train.csv', sep='\t', index = False, header = False)
data.iloc[int(len(data)*0.9):].to_csv('flair_dev.csv', sep='\t', index = False, header = False)
data_test.to_csv('flair_test.csv', sep='\t', index = False, header = False)

#train a model
from flair.data_fetcher import NLPTaskDataFetcher
from flair.embeddings import WordEmbeddings, FlairEmbeddings, DocumentRNNEmbeddings, CharLMEmbeddings
from flair.trainers import ModelTrainer
from flair.data import TaggedCorpus
from pathlib import Path

#es	Spanish	Spanish FastText embeddings (Wiki)
data_folder = Path('./')
corpus: TaggedCorpus = NLPTaskDataFetcher.load_classification_corpus(data_folder,                                                             
                                                                     test_file='flair_test.csv', 
                                                                     dev_file='flair_dev.csv', 
                                                                     train_file='flair_train.csv')
print(corpus)
print(len(corpus.train))


from hyperopt import hp
from flair.hyperparameter.param_selection import SearchSpace, Parameter



search_space = SearchSpace()
search_space.add(Parameter.EMBEDDINGS, hp.choice, options=[
    [ WordEmbeddings('ca') ], 
])
search_space.add(Parameter.HIDDEN_SIZE, hp.choice, options=[32, 64, 128])
search_space.add(Parameter.RNN_LAYERS, hp.choice, options=[1, 2])
search_space.add(Parameter.DROPOUT, hp.uniform, low=0.0, high=0.5)
search_space.add(Parameter.LEARNING_RATE, hp.choice, options=[0.05, 0.1, 0.15, 0.2])
search_space.add(Parameter.MINI_BATCH_SIZE, hp.choice, options=[8, 16, 32])


from flair.hyperparameter.param_selection import TextClassifierParamSelector, OptimizationValue

# create the parameter selector
param_selector = TextClassifierParamSelector(
    corpus, 
    False, 
    'resources/results', 
    'lstm',
    max_epochs=10, 
    training_runs=1,
    optimization_value=OptimizationValue.DEV_SCORE
)

# start the optimization
param_selector.optimize(search_space, max_evals=50)




#word_embeddings = [WordEmbeddings('es'), FlairEmbeddings('spanish-forward-fast'), FlairEmbeddings('spanish-backward-fast')]
#document_embeddings = DocumentRNNEmbeddings(word_embeddings, 
#                                            hidden_size=512, 
#                                            reproject_words=True, 
#                                            reproject_words_dimension=256,
#                                            bidirectional=True,
#                                            rnn_layers=2,
#                                            rnn_type='RNN_RELU')
#classifier = TextClassifier(document_embeddings, 
#                            label_dictionary=corpus.make_label_dictionary(), 
#                            multi_label=False)
#trainer = ModelTrainer(classifier, corpus)
#trainer.train('./', max_epochs=10)
#
#
##plot training curves
#from flair.visual.training_curves import Plotter
#plotter = Plotter()
#plotter.plot_training_curves('loss.tsv')
#plotter.plot_weights('weights.txt')
#
#
#from flair.models import TextClassifier
#from flair.data import Sentence
#classifier = TextClassifier.load_from_file('./best-model.pt')
#sentence = Sentence("Soluciones  venga ya Vendidos  Aprovechando el gran acontecimiento planetario del día de hoy rescatamos la portada de nuestro número de enero de est Esto no es un problema politico ya No os enterais de  esto es una REVOLUCION SOCIAL la habeis cagado")
#classifier.predict(sentence)
#print(sentence.labels)