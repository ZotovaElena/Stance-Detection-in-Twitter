# Stance-Detection-in-Twitter
Machine learning models to detect stance in short documents. 

This is a part of my master thesis Stance Detection in Twitter for Spanish and Catalan. It was written during my studies in the University of the Basque Country in 2017-2019. 

Stance is a way of thinking about some topic, expressed as a public opinion. In natural language processing, stance detection stands for automatically predicting whether the author of a statement is in favor of some event or person or against it. Usually, there is the third class, neutral, which represents the texts in which neither favor nor against stance can be detected. It is considered that stance detection is a part of opinion mining and it has much in common with the sentiment analysis. In sentiment analysis, systems determine whether a given text is positive, negative, or neutral, unlike this, in stance detection, systems are to determine author’s favorability towards a given target and the target even may be implicit. 

## Abstract

The majority of opinion mining tasks in natural language processing (NLP) have beenfocused on sentiment analysis of texts about products and services while there iscomparatively less research on automatic detection of political opinion. Almost allprevious research work has been done for English, while this thesis is focused on theautomatic detection of stance (whether he or she is favorable or not towards importantpolitical topic) from Twitter posts in Catalan, Spanish and English. The main objectiveof this work is to build and compare automatic stance detection systems using supervisedboth classic machine and deep learning techniques.  We also study the influence of textnormalization and perform experiments with different methods for word representationssuch as TF-IDF measures for unigrams, word embeddings, tweet embeddings, andcontextual character-based embeddings.  We obtain state-of-the-art results in the stancedetection task on the IberEval 2018 dataset. Our research shows that text normalizationand feature selection is important for the systems with unigram features, and does notaffect the performance when working with word vector representations. Classic methodssuch as unigrams and SVM classifer still outperform deep learning techniques, but seemto be prone to overfitting. The classifiers trained using word vector representations andthe neural network models encoded with contextual character-based vectors show greaterrobustness.

## Full master thesis

http://ixa.si.ehu.es/master/eu/node/2102

## References: 

Armand Joulin and douard Grave and Piotr Bojanowski and Tomas Mikolov, Bag of Tricks for Efficient Text Classification, 2016, https://dblp.org/rec/bib/journals/corr/JoulinGBM16 

Isabelle Augenstein,  Tim Rockt ̈aschel,  Andreas Vlachos,  and Kalina Bontcheva.  Stancedetection with bidirectional conditional encoding. InProceedings of the 2016 Conferenceon  Empirical  Methods  in  Natural  Language  Processing,  pages 876–885,  Austin,  Texas,2016.

Tomas Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean.  Efficient estimation of wordrepresentations in vector space.Computation and Language (cs.CL), 2013.

Saif Mohammad, Svetlana Kiritchenko, Parinaz Sobhani, Xiao-Dan Zhu, and Colin Cherry.A dataset for detecting stance in tweets.  InLREC, pages 3945–3952, 2016a.

M. Taulé, F. Rangel, M. A. Martí, and P. Rosso. Overview of the task on multimodal stancedetection  in  tweets  on  Catalan  1oct  referendum.   InIberEval  2018.  CEUR  WorkshopProceedings. CEUR-WS.org, pages 149–166, Sevilla, Spain, 2018.

Full list of references see in the [Master Thesis](http://ixa.si.ehu.es/master/eu/node/2102).
