import os
from typing import Generator
import gensim
from gensim.parsing.preprocessing import STOPWORDS
from gensim.utils import simple_preprocess
import nltk
from nltk.corpus import stopwords
STOPWORDS = stopwords.words('english')
STOPWORDS.extend(['from', 'subject', 're', 'edu', 'use'])
import spacy
from tqdm import tqdm

class WikiPreprocess:
    """Preprocessing class for wikipedia articles.
    """
    
    def preprocess_document(self, text: str, min_token_len) -> list:
        """Preprocesses a document. Tokenizes and removes stop words from document.
        
        Args:
            text (str): Wikipedia article

        Returns:
            list: List of preprocessed article tokens
        """
        tokens = gensim.utils.simple_preprocess(text, min_len=min_token_len, deacc=True)
        preprocessed_document = [token for token in tokens if token not in STOPWORDS]
        return preprocessed_document

    def create_dictionary(self, tokenized_documents: list) -> gensim.corpora.Dictionary:
        return gensim.corpora.Dictionary(tokenized_documents)

    def filtered_dictionary(self, preprocessed_documents: list, no_below, no_above) -> gensim.corpora.Dictionary:
        """Initializes a Dictionary and filters out tokens by their frequency.

        Args:
            preprocessed_documents (list): Preprocessed documents
            no_below (int):  # filter out tokens that appear in less than `no_below` documents
            no_above (int):  # filter out tokens that appear in more than `no_above` of corpus, a fraction


        Returns:
            gensim.corpora.Dictionary: Filtered dictionary
        """
        dictionary = self.create_dictionary(preprocessed_documents)
        dictionary.filter_extremes(no_below=no_below, no_above=no_above)
        return dictionary

    def articles_to_bow(self, dictionary: gensim.corpora.Dictionary, documents:list) -> list:
        """Converts articles into the bag-of-words (BoW) format = list of (token_id, token_count) tuples.

        Args:
            dictionary (gensim.corpora.Dictionary): Filtered dictionary
            documents (list): Preprocessed articles, list of (token_id, token_count) tuples for each article

        Returns:
            list: preprocessed articles in BoW format
        """
        preprocessed_bow_articles = [dictionary.doc2bow(d) for d in documents]
        return preprocessed_bow_articles

    def make_bigrams(self, preprocessed_documents:list):
        bigram = gensim.models.Phrases(preprocessed_documents, min_count=5, threshold=100)
        bigram_mod = gensim.models.phrases.Phraser(bigram)
        return [bigram_mod[doc] for doc in tqdm(preprocessed_documents)]

    def make_trigrams(self, preprocessed_documents:list):
        bigram = gensim.models.Phrases(preprocessed_documents, min_count=5, threshold=100)
        trigram = gensim.models.Phrases(bigram[preprocessed_documents], threshold=100)
        bigram_mod = gensim.models.phrases.Phraser(bigram)
        trigram_mod = gensim.models.phrases.Phraser(trigram)
        return [trigram_mod[bigram_mod[doc]] for doc in tqdm(preprocessed_documents)]

    def lemmatize(self, preprocessed_document:list, nlp, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
        document = nlp(" ".join(preprocessed_document))
        return [token.lemma_ for token in document if token.pos_ in allowed_postags]
