import numpy as np
from scipy.special import gammaln


class LDA:

    def __init__(self, corpus, num_topics, vocab_len, alpha=0.01, beta=0.01, **kwargs):
        """[summary]

        Args:
            num_topics ([type]): [description]
            num_docs ([type]): [description]
            vocab_len ([type]): [description]
            alpha (float, optional): [description]. Defaults to 0.01.
            beta (float, optional): [description]. Defaults to 0.01.
        """
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
    
    def initialize(self, documents):
        self.doc_topic_count = np.zeros([self.num_docs, self.num_topics])  # D x K, number of words in i'th document assigned to j'th topic, per document topic distribution
        self.topic_word_count = np.zeros([self.num_topics, self.vocab_len])  # K x W, number of times j'th word is assigned to i'th topic, per topic word distribution
        self.nz = np.zeros(self.num_topics)  # 1 x K, total word count for each topic
        self.per_doc_word_topic_assignment = [[0] * len(doc) for doc in documents]
        for d, doc in enumerate(documents):
            for w, bow in enumerate(doc):
                token_id, token_count = bow
                topic_idx = np.random.randint(self.num_topics)
                self.nz[topic_idx] += 1  # update count of total number of words assigned to j'th topic, topic count
                self.doc_topic_count[d, topic_idx] += 1  # update count for current word in current document assigned to j'th topic
                self.topic_word_count[topic_idx, token_id] += 1  # update count for current word assigned to j'th topic
                self.per_doc_word_topic_assignment[d][w] = topic_idx
    
    def fit(self, documents, max_iter):
        self.initialize(documents)
        self.perplexity_trace = []
        self.log_likelihood_trace = []
        for i in range(max_iter):
            for d, doc in enumerate(tqdm(documents)):
                for w, bow in enumerate(doc):
                    token_id, token_count = bow
                    topic_idx = self.per_doc_word_topic_assignment[d][w]
                    # decrement for current topic assignment
                    self.nz[topic_idx] -= 1  # update count of total number of words assigned to j'th topic
                    self.doc_topic_count[d, topic_idx] -= 1  # update count for current word in current document assigned to j'th topic
                    self.topic_word_count[topic_idx, token_id] -= 1  # update count for current word assigned to j'th topic
                    self.per_doc_word_topic_assignment[d][w] = topic_idx


                    # compute posterior p(z_dn = k|z_not_dn, w) , prob of current topic assignment given the word and all the other topic assignments                    
                    word_topic_count = self.topic_word_count[:, token_id]
                    topic_doc_count = self.doc_topic_count[d,:]
                    topic_doc_ratio = (topic_doc_count + self.alpha) / (len(doc) + (self.num_topics * self.alpha))
                    word_topic_ratio = (word_topic_count + self.beta) / (self.nz + (self.vocab_len * self.beta))
                    p_z_w = topic_doc_ratio * word_topic_ratio
                    full_cond_dist = p_z_w / np.sum(p_z_w)
                    new_topic_idx = np.random.multinomial(1, full_cond_dist).argmax()  # sample from multinomial dist

                    self.nz[new_topic_idx] += 1  # update count of total number of words assigned to j'th topic
                    self.doc_topic_count[d, new_topic_idx] += 1  # update count for current word in current document assigned to j'th topic
                    self.topic_word_count[new_topic_idx, token_id] += 1  # update count for current word assigned to j'th topic
                    self.per_doc_word_topic_assignment[d][w] = new_topic_idx
            
            ll = self.log_likelihood()
            print(f"ll:{ll}")
            self.log_likelihood_trace.append(ll)
            perplexity = np.exp(-ll / self.vocab_len)  # number of token, modify?
            self.perplexity_trace.append(perplexity)
            # if i % 10 == 1:
            print(f"iteration: {i}, log likelihood: {ll}, perplexity: {perplexity}")

            self.topic_word_count
        
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

    def top_n_topics(self, n):
        """Gets top n words associated with each to

        Args:
            n (_type_): _description_
        """
        tokens = []
        probs = []
        pass
    

# if __name__ == "__main__":
    # import bz2
    # import pickle
    # from pandas.core.frame import DataFrame
    # import gensim
    # from gensim.models import CoherenceModel
    # from tqdm import tqdm
    # import time
    # import spacy
    # import logging
    # import warnings
    # warnings.filterwarnings('ignore')
    # from wikipreprocess import WikiPreprocess

    # path = 'data/datapicklesoup.bz2'
    # with bz2.BZ2File(path, 'rb') as f:  #Use datacompression BZ2
    #     data = pickle.load(f)
    # # data = data[:]
    # data, _ = data
    # data = data[:1000]
    # print(len(data))
    # wiki_pp = WikiPreprocess()
    # preprocessed_data =  [wiki_pp.preprocess_document(text=d, min_token_len=4) for d in tqdm(data)]
    # data_words_bigrams = wiki_pp.make_bigrams(preprocessed_data)
    # nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
    # data_lemmatized = [wiki_pp.lemmatize(d, nlp, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']) for d in tqdm(data_words_bigrams)]
    # id2word_lemmatized = wiki_pp.filtered_dictionary(data_lemmatized, no_below=10, no_above=0.1)
    # print(len(id2word_lemmatized))
    # corpus_lemmatized_bow = [id2word_lemmatized.doc2bow(text) for text in tqdm(data_lemmatized)]
  
    # lda = LDA(corpus=corpus_lemmatized_bow, num_topics=10, vocab_len=len(id2word_lemmatized), alpha=0.01, beta=0.01)
    # lda.fit(documents=corpus_lemmatized_bow,max_iter=1)


    


