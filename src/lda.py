import re
import numpy as np
from typing import List
from scipy.special import gammaln
from tqdm import tqdm
import pyLDAvis


class LDA:

    def __init__(self, corpus: List[List[str]], num_topics: int, vocab_len: int, alpha: int = 0.01, beta: int = 0.01, **kwargs):
        self.corpus = corpus
        self.num_topics = num_topics
        self.num_docs = len(corpus)
        self.vocab_len = vocab_len
        self.alpha = np.array([alpha] * self.num_topics)
        self.alpha = alpha
        self.beta = beta
        self.doc_topic_count = np.zeros([self.num_docs, self.num_topics])  # D x K, number of words in i'th document assigned to j'th topic, per document topic distribution
        self.topic_word_count = np.zeros([self.num_topics, self.vocab_len])  # K x W, number of times j'th word is assigned to i'th topic, per topic word distribution
        self.nz = np.zeros(self.num_topics)  # 1 x K, total word count for each topic
        self.per_doc_word_topic_assignment = [[0] * len(doc) for doc in corpus]
    
    def initialize(self, documents: List[List[str]]):
        """Initializes the count matrices

        Args:
            documents (List[List[str]]): a list of bag of words for each document in the corpus
        """
        self.doc_topic_count = np.zeros([self.num_docs, self.num_topics])  # number of words in i'th document assigned to j'th topic, per document topic distribution
        self.topic_word_count = np.zeros([self.num_topics, self.vocab_len])  # number of times j'th term is assigned to i'th topic, per topic word distribution
        self.nz = np.zeros(self.num_topics)  # number of times a topic k is assigned in corpus
        self.per_doc_word_topic_assignment = [[0] * len(doc) for doc in documents]  # stores topic assignment for each term in each document
        for d, doc in enumerate(documents):
            for w, bow in enumerate(doc):
                token_id, token_count = bow
                topic_idx = np.random.randint(self.num_topics)
                # increment counts
                self.nz[topic_idx] += 1
                self.doc_topic_count[d, topic_idx] += 1
                self.topic_word_count[topic_idx, token_id] += 1
                self.per_doc_word_topic_assignment[d][w] = topic_idx
    
    def _sample(self):
        for d, doc in enumerate(tqdm(self.corpus)):
            for w, bow in enumerate(doc):
                token_id, token_count = bow
                topic_idx = self.per_doc_word_topic_assignment[d][w]

                # decrement for current topic assignment
                self.nz[topic_idx] -= 1
                self.doc_topic_count[d, topic_idx] -= 1
                self.topic_word_count[topic_idx, token_id] -= 1
                self.per_doc_word_topic_assignment[d][w] = topic_idx

                # compute full conditional distribution            
                word_topic_count = self.topic_word_count[:, token_id]
                topic_doc_count = self.doc_topic_count[d,:]
                topic_doc_ratio = (topic_doc_count + self.alpha) / (len(doc) + (self.num_topics * self.alpha))
                word_topic_ratio = (word_topic_count + self.beta) / (self.nz + (self.vocab_len * self.beta))
                p_z_w = topic_doc_ratio * word_topic_ratio
                full_cond_dist = p_z_w / np.sum(p_z_w)
                new_topic_idx = np.random.multinomial(1, full_cond_dist).argmax()  # sample from multinomial dist

                # increment count matrices
                self.nz[new_topic_idx] += 1  # update count of total number of words assigned to j'th topic
                self.doc_topic_count[d, new_topic_idx] += 1  # update count for current word in current document assigned to j'th topic
                self.topic_word_count[new_topic_idx, token_id] += 1  # update count for current word assigned to j'th topic
                self.per_doc_word_topic_assignment[d][w] = new_topic_idx

    def fit(self, documents: List[List[str]], burnin: int, max_iter: int):
        self.initialize(documents) 
        self.perplexity_trace = np.zeros(burnin + max_iter)
        self.log_likelihood_trace = np.zeros(burnin + max_iter)
        self.phi_trace = []
        self.theta_trace = []
        self.total_doc_topic_count = np.zeros([self.num_docs, self.num_topics])
        self.total_topic_word_count = np.zeros([self.num_topics, self.vocab_len]) 
        for i in range(burnin+max_iter):
            self._sample()

            # track log likelihood and perplexity
            ll = self.log_likelihood()
            self.log_likelihood_trace[i] = ll
            perplexity = np.exp(-ll / self.vocab_len)  # number of tokens, modify?
            self.perplexity_trace[i] = perplexity

            # accumulate counts for point estimates
            if not i % 10:
                print(f"iteration: {i} log_likelihood: {ll} perplexity: {perplexity}")
                self.total_doc_topic_count += self.doc_topic_count
                self.total_topic_word_count += self.topic_word_count

        
    def log_likelihood(self):
        ll = 0
        # log p(w|z)
        for k in range(self.num_topics):
            ll += np.sum(gammaln(self.topic_word_count[k,:] + self.beta)) - \
                gammaln(np.sum(self.topic_word_count[k,:] + self.beta))
            ll -= self.vocab_len * gammaln(self.beta) - gammaln(self.vocab_len*self.beta)

        # log p(z)
        for d, doc in enumerate(self.corpus):
            ll += np.sum(gammaln(self.doc_topic_count[d,:] + self.alpha)) - \
                gammaln(np.sum(self.doc_topic_count[d,:] + self.alpha))
            ll -= self.num_topics * gammaln(self.alpha) - gammaln(self.num_topics*self.alpha)

        return ll

    def get_phi(self):
        phi = self.total_topic_word_count + self.beta
        phi = phi / np.sum(phi, axis=1)[:,np.newaxis]
        return phi
        
    def get_theta(self):
        theta = self.total_doc_topic_count + self.alpha
        theta = theta / np.sum(theta, axis=1)[:, np.newaxis]
        return theta
    
    def top_topic_word_idx_mat(self, num_terms):
        topic_word_idx_sorted = np.argpartition(self.get_phi(), kth=range(-num_terms, 0), axis=-1)[:,-num_terms:]
        topic_word_idx_sorted = np.flip(topic_word_idx_sorted, axis=-1)
        return topic_word_idx_sorted
    
    def get_topics(self, vocab, k=10):
        topics = []
        for topic_idx, topic in enumerate(self.top_topic_word_idx_mat(num_terms=k)):
            words = []
            probs = []
            for word_idx in topic:
                word = vocab[word_idx]
                prob = self.get_phi()[topic_idx, word_idx]
                words.append(word)
                probs.append(prob)
            topics.append(dict(zip(words, probs)))
        return topics
    
    def print_topics(self, vocab, topn=10):
        for i, topic in enumerate(self.get_topics(vocab,topn)):
            topic = dict(sorted(topic.items(), key=lambda item: item[1], reverse=True))
            print(i,topic)

#     def plot_ldavis(self, dictionary):
#         doc_lengths = [len(doc) for doc in self.corpus]
#         tf = [dictionary.cfs[i] for i in range(len(dictionary))]
#         data = {'topic_term_dists': self.get_phi(), 
#             'doc_topic_dists': self.get_theta(),
#             'doc_lengths': doc_lengths,
#             'vocab': dictionary,
#             'term_frequency': tf}
#         vis_data = pyLDAvis.prepare(**data)
#         pyLDAvis.display(vis_data)


# if __name__ == "__main__":
#     import bz2
#     import pickle
#     from pandas.core.frame import DataFrame
#     import gensim
#     from gensim.models import CoherenceModel
#     from tqdm import tqdm
#     import time
#     import spacy
#     import logging
#     import warnings
#     warnings.filterwarnings('ignore')
#     from wikipreprocess import WikiPreprocess

#     path = 'data/datapicklesoup.bz2'
#     with bz2.BZ2File(path, 'rb') as f:  #Use datacompression BZ2
#         data = pickle.load(f)
#     # data = data[:]
#     data, _ = data
#     data = data[0:100]
#     # print(data)
#     print(len(data))
#     wiki_pp = WikiPreprocess()
#     preprocessed_data =  [wiki_pp.preprocess_document(text=d, min_token_len=4) for d in tqdm(data)]
#     data_words_bigrams = wiki_pp.make_bigrams(preprocessed_data)
#     nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
#     data_lemmatized = [wiki_pp.lemmatize(d, nlp, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']) for d in tqdm(data_words_bigrams)]
#     id2word_lemmatized = wiki_pp.create_dictionary(data_lemmatized)
#     print(id2word_lemmatized)
#     print(len(id2word_lemmatized))
#     corpus_lemmatized_bow = [id2word_lemmatized.doc2bow(text) for text in tqdm(data_lemmatized)]
  
#     lda = LDA(corpus=corpus_lemmatized_bow, num_topics=10, vocab_len=len(id2word_lemmatized), alpha=0.01, beta=0.01)
#     lda.fit(documents=corpus_lemmatized_bow,burnin=10,max_iter=1)
#     lda.print_topics(id2word_lemmatized, topn=10)
#     # lda.plot_ldavis(id2word_lemmatized)


    


