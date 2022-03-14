
'''
This file has been modified under GNU by:-
Duha Aldebakel
Yu Cao
Anthony Limon
Rui Zhang

We make two tweaks to the MFVI algorithm as described above. 

Firstly, we find it wasteful to repeat the E step indefinitely to find the perfect variational parameters $\phi$ and $\gamma$ under the assumption that $q(\beta_k)$ is correctly specified because we know that that assumption is certainly false at the start of the algorithm when $\lambda$ is assigned randomly. The focus should be to process as many documents as possible for an online algorithm and not be caught up with "perfect as the enemy of good". As the algorithm converges upon many passes, we will expect that the E step will not require many iterations anyway, as it will just be incremental changes at that point. Therefore, we changed the code to take the maximum number of E step iterations as a parameter, so that we can vary it from say 5 to 80. The results are shown in the results section of this write-up.

Secondly, we are motivated by Shen, Gao, and Ma [11] to find means to under-parameterize the MFVI model so that we can run experiments on early stopping, and in doing so, get hints about the true dimensionality of the underlying processes. This would answer the key question as to how many critical words are really needed to distinguish topics from the corpus, and bear light as to whether a human looking at the top 20 or 100 words can do this effectively. We do this under-parameterization by keeping the structure of the algorithm but no longer changing some of the $\lambda$ variational parameters effectively fixing them at baseline probabilities. We remove variational parameters from the most popular words first, as they are assumed to have the most commonality and hence less useful to distinguish topics. This parameterization removal does not change the size of the dataset nor the likelihood calculations so we can do an apples-to-apples comparison across under and over parameterized models. More details and results are available in the results section of this write-up.


'''

# onlineldavb.py: Package of functions for fitting Latent Dirichlet
# Allocation (LDA) with online variational Bayes (VB).
#
# Copyright (C) 2010  Matthew D. Hoffman
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import sys, re, time, string
import numpy as n
from scipy.special import gammaln, psi


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

n.random.seed(100000001)
meanchangethresh = 0.001

def dirichlet_expectation(alpha):
    """
    For a vector theta ~ Dir(alpha), computes E[log(theta)] given alpha.
    """
    if (len(alpha.shape) == 1):
        return(psi(alpha) - psi(n.sum(alpha)))
    return(psi(alpha) - psi(n.sum(alpha, 1))[:, n.newaxis])
  

class OnlineLDA:
    """
    Implements online VB for LDA as described in (Hoffman et al. 2010).
    """

    def __init__(self, vocab, K, D, alpha, eta, tau0, kappa):
        """
        Arguments:
        K: Number of topics
        vocab: A set of words to recognize. When analyzing documents, any word
           not in this set will be ignored.
        D: Total number of documents in the population. For a fixed corpus,
           this is the size of the corpus. In the truly online setting, this
           can be an estimate of the maximum number of documents that
           could ever be seen.
        alpha: Hyperparameter for prior on weight vectors theta
        eta: Hyperparameter for prior on topics beta
        tau0: A (positive) learning parameter that downweights early iterations
        kappa: Learning rate: exponential decay rate---should be between
             (0.5, 1.0] to guarantee asymptotic convergence.
        Note that if you pass the same set of D documents in every time and
        set kappa=0 this class can also be used to do batch VB.
        """
        self._vocab = dict()
        for word in vocab:
            #This will be done in preprocessing.
            #word = word.lower()
            #word = re.sub(r'[^a-z]', '', word)
            self._vocab[word] = len(self._vocab)

        self._K = K
        self._W = len(self._vocab)
        self._D = D
        self._alpha = alpha
        self._eta = eta
        self._tau0 = tau0 + 1
        self._kappa = kappa
        self._updatect = 0

        # Initialize the variational distribution q(beta|lambda)
        self._lambda = 1*n.random.gamma(100., 1./100., (self._K, self._W))
        self._Elogbeta = dirichlet_expectation(self._lambda)
        self._expElogbeta = n.exp(self._Elogbeta)
        
        self.maxEIter = 100
        self.vocabLimit=self._W
        self.sortedByFrequencies=None

    def do_e_step(self, wordids, wordcts):
        """
        Given a mini-batch of documents, estimates the parameters
        gamma controlling the variational distribution over the topic
        weights for each document in the mini-batch.
        Arguments:
        docs:  List of D documents. Each document must be represented
               as a string. (Word order is unimportant.) Any
               words not in the vocabulary will be ignored.
        Returns a tuple containing the estimated values of gamma,
        as well as sufficient statistics needed to update lambda.
        """

        batchD = len(wordids)

        # Initialize the variational distribution q(theta|gamma) for
        # the mini-batch
        gamma = 1*n.random.gamma(100., 1./100., (batchD, self._K))
        Elogtheta = dirichlet_expectation(gamma)
        expElogtheta = n.exp(Elogtheta)

        sstats = n.zeros(self._lambda.shape)
        # Now, for each document d update that document's gamma and phi
        it = 0
        meanchange = 0
        for d in range(0, batchD):
            # print(sum(wordcts[d]))
            # These are mostly just shorthand (but might help cache locality)
            ids = wordids[d]
            cts = wordcts[d]
            gammad = gamma[d, :]
            Elogthetad = Elogtheta[d, :]
            expElogthetad = expElogtheta[d, :]
            expElogbetad = self._expElogbeta[:, ids]                              
            # The optimal phi_{dwk} is proportional to 
            # expElogthetad_k * expElogbetad_w. phinorm is the normalizer.
            phinorm = n.dot(expElogthetad, expElogbetad) + 1e-100
            # Iterate between gamma and phi until convergence
            for it in range(0, self.maxEIter):
                lastgamma = gammad
                # We represent phi implicitly to save memory and time.
                # Substituting the value of the optimal phi back into
                # the update for gamma gives this update. Cf. Lee&Seung 2001.
                gammad = self._alpha + expElogthetad * \
                    n.dot(cts / phinorm, expElogbetad.T)                          #<----E step gamma update
                

                #print(gammad[:, n.newaxis])
                Elogthetad = dirichlet_expectation(gammad)                        # we use gammad to calculate Elogthetad
                expElogthetad = n.exp(Elogthetad)
                phinorm = n.dot(expElogthetad, expElogbetad) + 1e-100             #<----E step phi update
                # If gamma hasn't changed much, we're done.
                meanchange = n.mean(abs(gammad - lastgamma))
                if (meanchange < meanchangethresh):                               #<----E step ends when gamma stop changing
                    break
            gamma[d, :] = gammad
            # Contribution of document d to the expected sufficient
            # statistics for the M step.
            sstats[:, ids] += n.outer(expElogthetad.T, cts/phinorm)               #lambda shape is (k,w)

        # This step finishes computing the sufficient statistics for the
        # M step, so that
        # sstats[k, w] = \sum_d n_{dw} * phi_{dwk} 
        # = \sum_d n_{dw} * exp{Elogtheta_{dk} + Elogbeta_{kw}} / phinorm_{dw}.
        sstats = sstats * self._expElogbeta

        return((gamma, sstats))


    def update_lambda(self, wordids, wordcts):
        """
        First does an E step on the mini-batch given in wordids and
        wordcts, then uses the result of that E step to update the
        variational parameter matrix lambda.
        Arguments:
        docs:  List of D documents. Each document must be represented
               as a string. (Word order is unimportant.) Any
               words not in the vocabulary will be ignored.
        Returns gamma, the parameters to the variational distribution
        over the topic weights theta for the documents analyzed in this
        update.
        Also returns an estimate of the variational bound for the
        entire corpus for the OLD setting of lambda based on the
        documents passed in. This can be used as a (possibly very
        noisy) estimate of held-out likelihood.
        """

        # rhot will be between 0 and 1, and says how much to weight
        # the information we got from this mini-batch.
        rhot = pow(self._tau0 + self._updatect, -self._kappa)
        self._rhot = rhot
        # Do an E step to update gamma, phi | lambda for this
        # mini-batch. This also returns the information about phi that
        # we need to update lambda.
        (gamma, sstats) = self.do_e_step(wordids, wordcts)
        # Estimate held-out likelihood for current values of lambda.
        # bound = self.approx_bound(wordids, wordcts, gamma)
        bound=0.0 # Avoid calculating to save time.
        # Update lambda based on documents.
        self._lambda = self._lambda * (1-rhot) + \
            rhot * (self._eta + self._D * sstats / len(wordids))               #<----M step. eta="n",
                                                                               #     sstats captures summand from estep but across documents      
            
        # Code to under parameterize the model to explore early stopping times
        if self.vocabLimit!=self._W:
            self._lambda[:,self.sortedByFrequencies[:max(1,self._W-self.vocabLimit)]]=self._eta
        
        
        self._Elogbeta = dirichlet_expectation(self._lambda)                   # we use lambda to calculate expElogbeta
        self._expElogbeta = n.exp(self._Elogbeta)
        self._updatect += 1

        return(gamma, bound)

    def approx_bound(self, wordids, wordcts, gamma):
        """
        Estimates the variational bound over *all documents* using only
        the documents passed in as "docs." gamma is the set of parameters
        to the variational distribution q(theta) corresponding to the
        set of documents passed in.
        The output of this function is going to be noisy, but can be
        useful for assessing convergence.
        """

        # This is to handle the case where someone just hands us a single
        # document, not in a list.
        batchD = len(wordids)

        score = 0
        Elogtheta = dirichlet_expectation(gamma)
        expElogtheta = n.exp(Elogtheta)

        # E[log p(docs | theta, beta)]
        for d in range(0, batchD):
            ids = wordids[d]
            cts = n.array(wordcts[d])
            phinorm = n.zeros(len(ids))
            for i in range(0, len(ids)):
                temp = Elogtheta[d, :] + self._Elogbeta[:, ids[i]]
                tmax = max(temp)
                phinorm[i] = n.log(sum(n.exp(temp - tmax))) + tmax
            score += n.sum(cts * phinorm)
#             oldphinorm = phinorm
#             phinorm = n.dot(expElogtheta[d, :], self._expElogbeta[:, ids])
#             score += n.sum(cts * n.log(phinorm))

        # E[log p(theta | alpha) - log q(theta | gamma)]
        score += n.sum((self._alpha - gamma)*Elogtheta)
        score += n.sum(gammaln(gamma) - gammaln(self._alpha))
        score += sum(gammaln(self._alpha*self._K) - gammaln(n.sum(gamma, 1)))

        # Compensate for the subsampling of the population of documents
        score = score * self._D / len(wordids)

        # E[log p(beta | eta) - log q (beta | lambda)]
        score = score + n.sum((self._eta-self._lambda)*self._Elogbeta)
        score = score + n.sum(gammaln(self._lambda) - gammaln(self._eta))
        score = score + n.sum(gammaln(self._eta*self._W) - 
                              gammaln(n.sum(self._lambda, 1)))

        return(score)

from collections import defaultdict
eLoopTime=defaultdict(list, {5: [36, 58, 35, 30, 51, 47], 10: [35, 47, 43, 51, 55, 59], 20: [95, 68, 93, 75, 84, 81], 40: [84, 55, 86, 66, 103, 54], 80: [111, 50, 64, 63, 59, 85]})
eLoopBound=defaultdict(list, {5: [-111733851.43260013, -117781528.69408886, -123040409.73264208, -128633409.69968084, -121205953.4537439, -110966698.16985334], 10: [-129755490.7799989, -128921847.34090501, -117951450.05467737, -120393444.00248368, -120385338.58080256, -108159118.92671846], 20: [-124170809.46135023, -136015016.3987571, -106827136.22064555, -115562658.78387602, -124336169.61204082, -118594594.42351212], 40: [-114759190.73191096, -132424958.91197848, -133231030.76268373, -111836848.71508917, -131526004.52976249, -117476012.67518403], 80: [-117973922.65220611, -146022696.2883227, -117976902.21794659, -133024236.67739585, -120428154.47847381, -119522519.74053846]})


def run_onlineldavb(corpus: DataFrame, **kwargs):
   # Preprocessing articles
    logging.info('\n\n# Preprocessing articles')
    data = corpus.text.values
    wiki_pp = WikiPreprocess()
    print("Preprocessing...")
    starttime = time.time()
    logging.info('\n\n# Preprocessing articles (preprocessed_data)')
    preprocessed_data =  [wiki_pp.preprocess_document(text=d, min_token_len=kwargs['min_token_len']) for d in tqdm(data)]
    logging.info('\n\n# Preprocessing articles (preprocessed_data done)')
        
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
    

    # Fitting via LDA Online Variational Inference (Blei) library
    logging.info('\n\n# Fitting via LDA Online Variational Inference (Blei) library')

    from datetime import datetime
    import random

    K=20
    datasetsize=25
    sortedByFrequencies=sorted([t for t,f in id2word_lemmatized.cfs.items()],reverse=True)


    for maxE in [5]:
        random.shuffle(corpus_lemmatized)
        if True:
            start=datetime.now()

            D = 100000 #estimate of number of documents in the population
            S = 1000 #sample size of a batch
            alpha = 0.1
            eta = 0.01

            tau0 = 10.0
            kappa = 0.05


            '''
                Hyperparameter of Dirichlet priors
                    concentration parameter of Dirichlet prior, which smaller meaning more concentrated
                    we like it to be relatively concentrated since any topic usually uses a small number of key words
                    we allow one document to have more than one topic, but we want to penalize it when there are too many potential topics

                alpha: Hyperparameter for prior on weight vectors theta
                eta: Hyperparameter for prior on topics beta

                Learning rate parameters:
                tau0: A (positive) learning parameter that downweights early iterations
                kappa: Learning rate: exponential decay rate---should be between
                     (0.5, 1.0] to guarantee asymptotic convergence.


                For each document d, we would like to know the distributions of 
                    theta, the topic distribution of the document
                    z_n, the topic of n-th word in document
                    (so for each document, there is one theta and N z_n)

                but we only observe
                    w_n, the actual n-th word in document


            '''

            vocab=list(id2word_lemmatized.token2id.keys())
            model = OnlineLDA(vocab, K, D, alpha, eta, tau0, kappa)

            model.maxEIter=maxE


            bounds=[]
            bounds_h=[]
            times=[]
            lasttime=0
            for i in range(1000):
                j=i%datasetsize #We only have 50k documents, so we make another pass after 50
                batch=corpus_lemmatized[(j*S):((j+1)*S)]
                wordids = [[w for w,c in doc] for doc in batch]
                wordcts = [[c for w,c in doc] for doc in batch]
                if i==0:
                    holdids=wordids #First batch is the holdout
                    holdcts=wordcts #First batch is the holdout
                if j==0:
                    continue #Don't use holdout for learning

                (gamma, bound)=model.update_lambda(wordids, wordcts)
                (gamma_h, sstats_h) = model.do_e_step(holdids, holdcts)
                # Estimate held-out likelihood for current values of lambda.

                earlystoppingtime=(datetime.now()-start).seconds
                if earlystoppingtime<max(lasttime*1.1,lasttime+5):
                    continue

                lasttime=earlystoppingtime
                bound_h = model.approx_bound(holdids, holdcts, gamma_h)
                #bound is the evidence lower bound (ELBO)

                logging.info('elasped {}s bound_h {}'.format(earlystoppingtime,bound_h))

                bounds.append(bound)
                bounds_h.append(bound_h)
                times.append(earlystoppingtime)

                if len(bounds_h)>3:
                    if bounds_h[-1]*(1+10**-5) < bounds_h[-3]*0.5+bounds_h[-2]*0.5:
                        break
            eLoopTime[maxE].append(times[-1])
            eLoopBound[maxE].append(bound_h)
            logging.info(str(eLoopTime))
            logging.info(str(eLoopBound))


    logging.info('Time taken = {:.0f} minutes'.format((time.time()-starttime)/60.0))
    

  
