# -*- coding: utf-8 -*-
"""
Spyder Editor

lr_text_classifier.py

see p231+ from Abshishek's book

This is a temporary script file.
"""
import numpy as np
import os
os.getcwd()
wd = '/Users/paulpaul/bbc/'
wd = '/Users/ftk/PYCODE/bbc/'

import pandas as pd
from nltk.tokenize import word_tokenize
from sklearn import linear_model, metrics, model_selection
from sklearn.feature_extraction.text import CountVectorizer
import time

#if __name__ == "__main__":
d = pd.read_csv(wd+'bbc_summ.csv')

ix = np.logical_or(  d.lblnum.values==0 ,  d.lblnum.values==1 )
d = d[ix].reset_index(drop=True)

d['kfold']=-1
d=d.sample(frac=1).reset_index(drop=True)

# set the text to be used as features: articles (txt_x) or summary (txt_y)
d['feature_txt'] = d['txt_y']

# initialize the kfold class from model_selection module
kf = model_selection.StratifiedKFold(n_splits=5)

# fill the new kfold columns
for f , (t_,v_) in enumerate(kf.split(X=d,y=d.lblnum.values)):
    d.loc[ v_, 'kfold'] = f
#d.kfold.values

res = {}    
for fold_ix in range(5):
    start = time.time()
    # temp dfs for train and test
    train_df = d[ d.kfold != fold_ix ].reset_index(drop=True)
    test_df  = d[ d.kfold == fold_ix ].reset_index(drop=True)
    
    # init countVectorizer with NLTK's word_tokenizer
    count_vec = CountVectorizer(
            tokenizer=word_tokenize,
            token_pattern=None )
    count_vec.fit(train_df.feature_txt)
        
    # transform train and test sets
    xtrain = count_vec.transform(train_df.feature_txt)
    xtest  = count_vec.transform(test_df.feature_txt)
    
    # init LR model
    model = linear_model.LogisticRegression()
    model.fit( xtrain, train_df.lblnum)
    
    preds = model.predict( xtest )
    
    accuracy = metrics.accuracy_score( test_df.lblnum, preds )
    runtime = time.time() - start
    res[fold_ix] = [accuracy, runtime]
    print(f'Fold:{fold_ix}  has accuracy={accuracy}')
#%%    
print(res)
#%%
