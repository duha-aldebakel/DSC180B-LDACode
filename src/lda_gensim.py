try:
    from wikipreprocess import WikiPreprocess
except ImportError:
    from src.wikipreprocess import WikiPreprocess

from pandas.core.frame import DataFrame
from helpers.log import setUpNewLogFile
import gensim
from gensim.models import CoherenceModel
from tqdm import tqdm
import time
import spacy
import logging
import warnings
warnings.filterwarnings('ignore')

def run_lda_gensim(corpus: DataFrame, **kwargs):
   # Preprocessing articles
    logging.info('\n\n# Preprocessing articles')
    data = corpus.text.values
    wiki_pp = WikiPreprocess()
    print("Preprocessing...")
    starttime = time.time()
    preprocessed_data =  [wiki_pp.preprocess_document(text=d, min_token_len=kwargs['min_token_len']) for d in tqdm(data)]
    
    # Adding bigrams
    logging.info('\n\n# Adding bigrams')
    print("Creating bigrams... ")
    data_words_bigrams = wiki_pp.make_bigrams(preprocessed_data)


    # Lemmatization
    logging.info('\n\n# Lemmatization')
    print("Lemmatizing data and creating dictionary from bigrams... ")
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
    data_lemmatized = [wiki_pp.lemmatize(d, nlp, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']) for d in tqdm(data_words_bigrams)]
    id2word_lemmatized = wiki_pp.filtered_dictionary(data_lemmatized, no_below=10, no_above=0.1)
  

    # Creating bag of words frequencies
    logging.info('\n\n# Creating bag of words frequencies.')
    print("Creating bag of words frequencies... ")
    corpus_lemmatized = [id2word_lemmatized.doc2bow(text) for text in tqdm(data_lemmatized)]
    dictwords=set(id2word_lemmatized.values())
    data_lemmatized_filtered=[[w for w in article if w in dictwords] for article in data_lemmatized]
    

    # Fitting via LDA Variational Inference (Gensim) library
    logging.info('\n\n# Fitting via LDA Variational Inference (Gensim) library')
    ## With Lemmatization
    logging.info('\n\n## Fitting with lemmatization')
    setUpNewLogFile('gensim_lem.log')
    print("Fitting model... ", end="")
    starttime = time.time()
    lda_model_lemmatized = gensim.models.ldamulticore.LdaMulticore(
        corpus=corpus_lemmatized, 
        id2word=id2word_lemmatized, 
        num_topics=kwargs['num_topics'], 
        random_state=100, 
        eval_every=kwargs["eval_every"], 
        chunksize=kwargs["chunk"], 
        passes=kwargs["passes"], 
        alpha='symmetric', 
        per_word_topics=True, 
        workers=kwargs["workers"]
    )
    print("Done.")
    logging.info('Time taken = {:.0f} minutes'.format((time.time()-starttime)/60.0))
    
    ## Evaluate model
    print("Evaluating model...")
    print('\nLog Likelihood(per-word ELBO): ', lda_model_lemmatized.log_perplexity(corpus_lemmatized))
    coherence_model_lda_lemmatized = CoherenceModel(
        model=lda_model_lemmatized, 
        texts=data_lemmatized_filtered, 
        dictionary=id2word_lemmatized, 
        coherence='c_v'
    )
    coherence_lda_lemmatized = coherence_model_lda_lemmatized.get_coherence()
    print('\nCoherence Score: ', coherence_lda_lemmatized)

    