#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd

def lemmatize(tweets, lemma_table):
	
	#We create lemma dict
	
	flex_all = list(lemma_table['Flex'])
	lemma_all = list(lemma_table['Lemma'])
	dic_lemmas = {}

	for flex,lemma in zip(flex_all, lemma_all):
		dic_lemmas[flex] = lemma
		
	#Create a copy of the table and include a new column
	
	lemmas = tweets.copy()
	lemmas = lemmas.fillna("")
	lemmas['text_lemma'] = ''
	
	for i, l in enumerate(lemmas['Tweet']):
		tweet_lemmas = []
		
		#Get lowercase tweet and split tokens
		tokens = l.lower().split()
		
		#If the word is in the dictionary include the lemma in the text_lemma
		for t in tokens:
			if t in dic_lemmas:
				tweet_lemmas.append(dic_lemmas[t])
			else:
				tweet_lemmas.append(t)
		lemmas['text_lemma'][i] = ' '.join(tweet_lemmas)
		print(i, tweet_lemmas)
		print(lemmas['text_lemma'][i])
	#Return the new dataframe with new lemma column
	return lemmas
