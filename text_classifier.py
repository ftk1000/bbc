# -*- coding: utf-8 -*-
"""
Spyder Editor

text_classifier.py

see p231+ from Abshishek's book

This is a temporary script file.
"""
import numpy as np
import os
os.getcwd()
wd = '/Users/paulpaul/bbc/'

import pandas as pd
from nltk.tokenize import word_tokenize
from sklearn import  metrics, model_selection, naive_bayes ,linear_model
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import time


#MODELS = ['LR', 'NB']
##MODELS = [ 'LR']
#
#VECTORIZERS = ['count_vec', 'tfidef_vec']; 
#TEXT = ['article', 'summary'] 


#if __name__ == "__main__":
d = pd.read_csv(wd+'bbc_summ.csv')

#ix = np.logical_or(  d.lblnum.values==0 ,  d.lblnum.values==1 )
#ix =   d.lblnum.values <= 2
#d = d[ix].reset_index(drop=True)

d['kfold']=-1
d=d.sample(frac=1).reset_index(drop=True)

# initialize the kfold class from model_selection module
kf = model_selection.StratifiedKFold(n_splits=5)

# fill the new kfold columns
for f , (t_,v_) in enumerate(kf.split(X=d,y=d.lblnum.values)):
    d.loc[ v_, 'kfold'] = f
#d.kfold.values
#resdf = pd.DataFrame(columns=['model','vectorizer','text', 'accuracy','time','fold'])


def run_cv_folds(ModelStr, Text, VectorizerStr):    
    # set the text to be used as features: articles (txt_x) or summary (txt_y)
    if Text  == 'article':
        d['feature_txt'] = d['txt_x'] 
    elif Text == 'summary':    
        d['feature_txt'] = d['txt_y']
        
    accuracy_list=[]
    time_list=[]
    
    for fold_ix in range(5):
        start = time.time()
        # temp dfs for train and test
        train_df = d[ d.kfold != fold_ix ].reset_index(drop=True)
        test_df  = d[ d.kfold == fold_ix ].reset_index(drop=True)
        
        if VectorizerStr == 'count_vec':
            # init countVectorizer with NLTK's word_tokenizer
            vectorizer = CountVectorizer(
                    tokenizer=word_tokenize,
                    ngram_range = (1,2),
                    token_pattern=None )
        elif  VectorizerStr == 'tfidf_vec':
            vectorizer = TfidfVectorizer(
                    tokenizer=word_tokenize,
                    ngram_range = (1,2),
                    token_pattern=None )
            
        vectorizer.fit(train_df.feature_txt)
                
        # transform train and test sets
        xtrain = vectorizer.transform(train_df.feature_txt)
        xtest  = vectorizer.transform(test_df.feature_txt)
                
        # init LR model
        if ModelStr=='LR':
            model = linear_model.LogisticRegression()
        elif ModelStr=='NB':
            model = naive_bayes.MultinomialNB()
            
        model.fit( xtrain, train_df.lblnum)
        
        preds = model.predict( xtest )
        
        accuracy = metrics.accuracy_score( test_df.lblnum, preds )
        accuracy_list.append(round(accuracy,2))
        
        runtime = round( time.time() - start, 2)
        time_list.append(runtime)
        
    #    res = {'model':MODEL,'vectorizer':VECTORIZER,'text':TEXT, 
    #           'accuracy': round(accuracy,2) ,
    #           'time': round(runtime,1) ,
    #           'fold':fold_ix}
    #    
    #    resdf = resdf.append(res, ignore_index=True)
        
        print(f'Fold:{fold_ix}  has accuracy={accuracy}')
        
        
    res = {'model':ModelStr,
       'vectorizer':VectorizerStr,
       'text':Text, 
       'acc_mean': round( np.mean(accuracy_list), 2), 
       'time_mean': round( np.mean(time_list), 2),
       'accuracy': accuracy_list ,
       'time': time_list }
    
    return res
#%%    
MODELS = ['LR', 'NB']
VECTORIZERS = ['count_vec', 'tfidf_vec']; 
#TEXT = ['article', 'summary'] 
TEXT = ['article'] 
    
resdf = pd.DataFrame(columns=['model','vectorizer','text', 
                              'acc_mean', 'time_mean',
                              'accuracy','time'])
    
for model in MODELS:   
    for vect in VECTORIZERS:
        for text in TEXT:
            res = run_cv_folds(ModelStr=model, Text=text, VectorizerStr=vect)
            resdf = resdf.append(res, ignore_index=True)
    
print(resdf)
#%%