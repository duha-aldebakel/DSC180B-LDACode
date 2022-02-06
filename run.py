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
import logging.handlers
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

# Loading data pickle file
logging.info('\n\n# Loading data pickle file '+fn)
    
with bz2.BZ2File(fn, 'rb') as f:  #Use datacompression BZ2
    data= pickle.load(f)   
df=pd.DataFrame({'text':data[0],'title':data[1]})

#remove zero length articles
articlelen=df.text.apply(len)
df=df[articlelen>10]

logging.info(str(df))

# Tokenizing articles
logging.info('\n\n# Tokenizing articles')

def sent_to_words(sentences):
   for sentence in sentences:
      yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))
data_words = list(sent_to_words(df.text))
id2word = corpora.Dictionary(data_words)
texts = data_words
corpus = [id2word.doc2bow(text) for text in texts]

# Removing stop words and adding bigrams
logging.info('\n\n# Removing stop words and adding bigrams')

bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100)
trigram = gensim.models.Phrases(bigram[data_words], threshold=100)
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)
def remove_stopwords(texts):
   return [[word for word in simple_preprocess(str(doc)) 
   if word not in stop_words] for doc in texts]
def make_bigrams(texts):
   return [bigram_mod[doc] for doc in texts]
def make_trigrams(texts):
   [trigram_mod[bigram_mod[doc]] for doc in texts]

data_words_nostops = remove_stopwords(data_words)
data_words_bigrams = make_bigrams(data_words_nostops)
logging.info("data_words_nostops[0][:40]")
logging.info(str(data_words_nostops[0][:40]))
logging.info("data_words_bigrams[0][:40]")
logging.info(str(data_words_bigrams[0][:40]))

# Lemmatization
logging.info('\n\n# Lemmatization')

doc=nlp(" ".join(data_words_bigrams[0]))
tags=[]
for w in doc:
    if not w.pos_ in tags:
        logging.info('Lemmatization Example {} -> {}'.format(w.lemma_,w.pos_))
        tags.append(w.pos_)
def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
   texts_out = []
   for sent in texts:
      doc = nlp(" ".join(sent))
      texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
   return texts_out

def lemmatization2(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
   texts_out = []
   for sent in texts:
      doc = nlp(sent)
      texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
   return texts_out

#bigrams, then lemmatize, then remove stop words
data_lemmatized = make_bigrams(data_words)

nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
data_lemmatized = lemmatization(data_lemmatized, allowed_postags=[
   'NOUN', 'ADJ', 'VERB', 'ADV'
])
data_lemmatized = remove_stopwords(data_lemmatized)

logging.info("data_lemmatized[0][:40]")
logging.info(str(data_lemmatized[0][:40]))
logging.info("data_words_bigrams[0][:40]")
logging.info(str(data_words_bigrams[0][:40]))

# Creating bag of words frequencies
logging.info('\n\n# Creating bag of words frequencies')

id2word = corpora.Dictionary(data_words_bigrams)
texts = data_words_bigrams
corpus = [id2word.doc2bow(text) for text in texts]

logging.info(f'No lemmatization -- Number of unique tokens: {len(id2word)}')
logging.info(f'No lemmatization -- Number of documents: {len(corpus)}')

id2word_lemmatized = corpora.Dictionary(data_lemmatized)

# Filter out words that occur less than X documents, 
# or more than X% of the documents.
id2word_lemmatized.filter_extremes(no_below=10, no_above=0.1)

corpus_lemmatized = [id2word_lemmatized.doc2bow(text) for text in data_lemmatized]

logging.info(f'lemmatization -- Number of unique tokens: {len(id2word_lemmatized)}')
logging.info(f'lemmatization -- Number of documents: {len(corpus_lemmatized)}')

dictwords=set(id2word_lemmatized.values())
data_lemmatized_filtered=[[w for w in article if w in dictwords] for article in data_lemmatized]

# Fitting via LDA Variational Inference (Gensim) library
logging.info('\n\n# Fitting via LDA Variational Inference (Gensim) library')


def setUpNewLogFile(LOG_FILENAME):

    #logging.basicConfig(filename=LOG_FILENAME,
    #                    format="%(asctime)s:%(levelname)s:%(message)s",
    #                    level=logging.INFO)

    my_logger = logging.getLogger()
    my_logger.setLevel(logging.INFO)
    my_logger.handlers.clear()
    handlers = my_logger.handlers[:]
    for handler in handlers:
        handler.close()
        my_logger.removeHandler(handler)

    # Check if log exists and should therefore be rolled
    needRoll = os.path.isfile(LOG_FILENAME)

    # Add the log message handler to the logger
    handler = logging.handlers.RotatingFileHandler(LOG_FILENAME, backupCount=50)

    my_logger.addHandler(handler)

    # This is a stale log, so roll it
    if needRoll:    
        # Add timestamp
        my_logger.debug('\n---------\nLog closed on %s.\n---------\n' % time.asctime())

        # Roll over on application start
        my_logger.handlers[0].doRollover()

## Without Lemmatization
logging.info('\n\n## Without Lemmatization')
starttime=time.time()
setUpNewLogFile('gensim_nolem.log')

#Old code
#lda_model = gensim.models.ldamodel.LdaModel(

#New code uses multicore which runs works in parallel for each CPU core.
lda_model = gensim.models.ldamulticore.LdaMulticore(
   corpus=corpus, id2word=id2word, num_topics=20, random_state=100, 
   eval_every=1000, chunksize=200, passes=20, alpha='symmetric', per_word_topics=True
)
logging.info('Time taken = {:.0f} minutes'.format((time.time()-starttime)/60.0))


p = re.compile(r"(-*\d+\.\d+) per-word .* (\d+\.\d+) perplexity")
matches = [p.findall(l) for l in open('gensim_nolem.log')]
matches = [m for m in matches if len(m) > 0]
tuples = [t[0] for t in matches]
perplexity = [float(t[1]) for t in tuples]
liklihood = [float(t[0]) for t in tuples]
iter = list(range(0,len(tuples)*10,10))
plt.plot(iter[:-1],liklihood[:-1],c="black")
plt.ylabel("log likelihood")
plt.xlabel("iteration")
plt.title("Topic Model Convergence")
plt.grid()
plt.savefig('images/NoLemConvergenceLikelihood.eps', format='eps')
plt.savefig('images/NoLemConvergenceLikelihood.png')


plt.plot(iter[:-1],perplexity[:-1],c="black")
plt.ylabel("Perplexity")
plt.xlabel("iteration")
plt.title("Topic Model Convergence")
plt.grid()
plt.savefig('images/NoLemConvergencePerplexity.eps', format='eps')
plt.savefig('images/NoLemConvergencePerplexity.png')
logging.info('Note: Perplexity estimate based on a held-out corpus of 4 documents')

## With Lemmatization
logging.info('\n\n## With Lemmatization')
starttime=time.time()
setUpNewLogFile('gensim_lem.log')

#Old code
#lda_model_lemmatized = gensim.models.ldamodel.LdaModel(

#New code uses multicore which runs works in parallel for each CPU core.
lda_model_lemmatized = gensim.models.ldamulticore.LdaMulticore(
   corpus=corpus_lemmatized, id2word=id2word_lemmatized, num_topics=20, random_state=100, 
   eval_every=1000, chunksize=1000, passes=10, alpha='symmetric', per_word_topics=True
)
logging.info('Time taken = {:.0f} minutes'.format((time.time()-starttime)/60.0))

p = re.compile(r"(-*\d+\.\d+) per-word .* (\d+\.\d+) perplexity")
matches = [p.findall(l) for l in open('gensim_lem.log')]
matches = [m for m in matches if len(m) > 0]
tuples = [t[0] for t in matches]
perplexity = [float(t[1]) for t in tuples]
liklihood = [float(t[0]) for t in tuples]
iter = list(range(0,len(tuples)*10,10))
plt.plot(iter[:-1],liklihood[:-1],c="black")
plt.ylabel("log likelihood")
plt.xlabel("iteration")
plt.title("Topic Model Convergence")
plt.grid()
plt.savefig('images/LemConvergenceLikelihood.eps', format='eps')
plt.savefig('images/LemConvergenceLikelihood.png')


plt.plot(iter[:-1],perplexity[:-1],c="black")
plt.ylabel("Perplexity")
plt.xlabel("iteration")
plt.title("Topic Model Convergence")
plt.grid()
plt.savefig('images/LemConvergencePerplexity.eps', format='eps')
plt.savefig('images/LemConvergencePerplexity.png')
logging.info('Note: Log likelihood is per-word ELBO')
logging.info('Note: Perplexity estimate based on a held-out corpus of 4 documents')

