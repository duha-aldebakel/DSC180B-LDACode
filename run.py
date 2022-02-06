import re
import numpy as np
import pandas as pd
from pprint import pprint
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
import spacy
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis

import matplotlib.pyplot as plt
import nltk
nltk.download('stopwords')
#python -m spacy download en_core_web_sm
#python -m spacy download en_core_web_md
nlp = spacy.load('en_core_web_md', disable=['parser', 'ner'])
from nltk.corpus import stopwords

stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])
from sklearn.datasets import fetch_20newsgroups

import pickle
import bz2
import time
import logging
import os


import logging

logging.basicConfig(filename='app.log', 
                    filemode='w',
                    format='%(asctime)s - %(name)s %(levelname)s %(message)s', 
                    datefmt='%d-%b-%y %H:%M:%S',
                    level=logging.DEBUG)

import sys
logging.info('Number of arguments: {} arguments.'.format(len(sys.argv)))
logging.info( 'Argument List: {}'.format(str(sys.argv)))
testmode=False
if sys.argv[-1].lower().strip()=='test':
    logging.info('App started in test mode. Using test data (generated with just 2 features. See makeTestData.py)')
    testmode=True
else:
    logging.info('App started in full mode.')


fn='data/datapicklesoup.bz2'
if testmode:
    fn='test/testdata/testdatapickle.bz2'
    
logging.info('Loading data pickle file '+fn)
    
with bz2.BZ2File(fn, 'rb') as f:  #Use datacompression BZ2
    data= pickle.load(f)   
df=pd.DataFrame({'text':data[0],'title':data[1]})

#remove zero length articles
articlelen=df.text.apply(len)
df=df[articlelen>10]

logging.info(str(df))

