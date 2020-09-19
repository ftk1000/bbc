# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os 
import glob
os.getcwd()
wd = '/Users/paulpaul/bbc/'
dira = wd+'BBC News Summary/News Articles/'
dirs = wd+'BBC News Summary/Summaries/'

os.chdir(wd)
os.getcwd()

'''
# set default encoding
import sys
reload(sys)
sys.setdefaultencoding("ISO-8859-1")

import codecs
with codecs.open('unicode.rst', encoding='utf-8') as f:
    for line in f:
        print repr(line)
'utf-8'
"ISO-8859-1"
'''

#!ls -ltra "BBC News Summary"
# get text from sub-DIRECTORies of dirdir
import numpy as np
import pandas as pd
import codecs
def get_texts(dirdir):
    txt = []
    fn=[]
    lbl=[]
    lblnum=[]
    leng = []
    #business	entertainment	politics 	sport		tech
    subdirs =['entertainment/', 'business/','politics/', 'sport/', 'tech/']
    for i in range(len(subdirs)):
        subdir = subdirs[i]
        folder_path = dirdir+subdir
        for filename in glob.glob(os.path.join(folder_path, '*.txt')):
#            with open(filename, 'rb') as f:
            with codecs.open( filename, encoding="ISO-8859-1") as f:
                text = f.read()
                txt.append(text)
                lbl.append(subdir[0])
                lblnum.append(i)
                fn.append(filename.split('/')[-1].split('.')[0])
                print (filename)
                leng.append(len(text))
                           
    d = pd.DataFrame(zip(fn,lblnum, lbl, txt, leng),
                     columns=['fn', 'lblnum', 'lbl', 'txt','leng'])
    return d
#%%
d=get_texts(dirdir=dira)
d2=get_texts(dirdir=dirs)
#%%
d.columns
dd=pd.merge(d, d2,  how='inner', on=['fn', 'lblnum', 'lbl'])
out_fn = wd+'bbc_summ.csv'
dd.to_csv(out_fn)