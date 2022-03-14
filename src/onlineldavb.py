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
            logging.info('runtimes under different maximum E step iterations (cummulative from many runs) are: '+str(eLoopTime))
            logging.info('ELBO under different maximum E step iterations (cummulative from many runs) are: '+str(eLoopBound))
            logging.info('Please see notebooks for plots')

    logging.info('\n\nNow doing Under parameterizaion/Early Stopping experiment..')
    from datetime import datetime
    import random
    from collections import defaultdict

    optimalStopping=defaultdict(list, {(49, 1000): [28, 26, 25, 25, 37, 24, 23, 25, 25, 22, 24, 29, 25, 26, 25, 25, 25], (49, 2500): [25, 22, 27, 25, 25, 25, 24, 24, 26, 24, 37, 31, 24, 25, 26, 37, 33], (49, 5000): [72, 33, 39, 21, 47, 54, 31, 49, 41, 27, 60, 56, 36, 43, 58, 50, 54], (49, 7500): [141, 56, 46, 79, 47, 65, 56, 59, 45, 52, 80, 93, 36, 44, 64, 74, 60], (49, 10000): [90, 69, 69, 102, 42, 53, 52, 80, 83, 44, 64, 66, 35, 63, 104, 76, 72], (49, 12500): [79, 129, 66, 132, 50, 53, 67, 42, 82, 47, 69, 109, 35, 81, 86, 67, 54], (49, 17500): [85, 72, 108, 199, 74, 74, 64, 133, 82, 59, 70, 71, 30, 54, 78, 51, 36], (49, 25000): [46, 46, 132, 177, 36, 72, 40, 45, 77, 41, 38, 70, 29, 88, 82, 49, 73], (24, 1000): [28, 25, 23, 26, 26, 27, 24, 32, 26, 25, 23, 23, 26, 23, 25], (24, 2500): [25, 21, 27, 23, 25, 21, 25, 22, 32, 35, 24, 24, 24, 25, 26], (24, 5000): [28, 55, 50, 49, 41, 49, 66, 45, 47, 59, 60, 25, 50, 60, 81], (24, 7500): [58, 58, 52, 44, 90, 51, 86, 54, 61, 72, 53, 60, 50, 40, 44], (24, 10000): [145, 86, 70, 57, 95, 70, 45, 70, 58, 72, 47, 54, 48, 38, 43], (24, 12500): [116, 44, 57, 61, 87, 25, 49, 43, 59, 71, 51, 53, 49, 39, 44], (24, 17500): [105, 41, 59, 51, 92, 31, 46, 40, 53, 77, 66, 47, 58, 41, 62], (24, 25000): [51, 38, 55, 57, 34, 77, 39, 58, 53, 72, 29, 47, 58, 46, 54]})
    boundsh=defaultdict(list, {(49, 1000): [-998194257.16, -1095456596.58, -1100399814.2, -1029909183.66, -981458399.08, -903910300.52, -1102162092.2, -1160587733.75, -1076599139.88, -1052261098.95, -965457916.98, -1178935419.42, -983501853.57, -957531625.96, -1154346182.93, -1051951686.17, -1042566358.64], (49, 2500): [-665666849.5, -719763935.52, -731620668.44, -687558590.63, -660973095.7, -610420930.45, -732937523.79, -771285165.2, -718297372.6, -697108066.67, -648831843.7, -780586681.9, -655100105.8, -634984758.53, -763871978.91, -696454849.88, -701043744.76], (49, 5000): [-419607059.68, -463499888.28, -473984839.69, -440411852.15, -421756218.69, -390860733.83, -468692155.37, -491188625.63, -458904364.01, -445412888.61, -415679633.0, -492610164.07, -412043615.76, -406004812.7, -487688760.82, -436378845.72, -444174653.82], (49, 7500): [-313451185.16, -349914160.78, -351496716.56, -332178324.49, -316071073.55, -296508916.53, -348425071.7, -369345602.91, -339886507.53, -332348100.66, -309702249.56, -371624796.26, -312266309.63, -305929642.55, -365674875.44, -325919534.52, -337395362.39], (49, 10000): [-253381669.23, -285237898.9, -283415976.65, -268567647.79, -257398119.96, -241925088.44, -280337975.13, -301295905.72, -276085824.53, -271390287.61, -250078172.75, -303820706.01, -254351714.33, -248589584.49, -289296695.87, -263713816.98, -270059938.5], (49, 12500): [-211749383.31, -233941880.14, -236564246.26, -218724536.15, -215864683.01, -198976275.66, -234468706.36, -251486438.45, -229160364.69, -224620662.09, -206789141.5, -249526378.86, -212679117.04, -207966048.18, -241961178.56, -220177375.57, -225920026.5], (49, 17500): [-162998427.61, -178125247.19, -177715543.54, -166037349.51, -163864490.63, -151177561.37, -176701850.46, -189896192.32, -177425273.61, -173379475.35, -158401218.56, -189789317.1, -163971285.34, -158279688.83, -186105854.24, -166498616.96, -171464885.08], (49, 25000): [-126141984.68, -137410075.78, -135109696.92, -125637042.01, -125286014.7, -111500899.45, -136662757.28, -147891742.35, -134649074.98, -132338630.06, -124552127.51, -144650430.74, -126733335.91, -117700163.94, -141085515.15, -128430643.51, -128442679.24], (24, 1000): [-1047066521.48, -1059318107.29, -1024938520.62, -998660237.26, -1198712524.03, -1107968994.61, -1104506349.86, -1110338255.15, -1086691525.84, -1063666118.37, -1121020661.94, -845643713.98, -912724713.6, -1118647747.49, -899014079.21], (24, 2500): [-697684131.99, -703759891.28, -682433386.39, -664655460.43, -804248851.55, -740584950.37, -731378554.02, -739477885.61, -714687859.73, -707708467.42, -743717869.98, -562371163.63, -602675871.63, -744016095.65, -588504587.7], (24, 5000): [-440641225.11, -453375292.97, -430877066.3, -422867628.36, -510132046.03, -477394955.64, -467215423.23, -467476332.39, -449765594.06, -448622603.96, -473515478.15, -360241910.49, -385159273.42, -473489990.64, -374986746.94], (24, 7500): [-329768828.97, -338553341.5, -323980951.42, -314746492.4, -382080818.25, -357047119.93, -349111593.83, -346282613.04, -335998421.39, -333720962.02, -354098447.33, -267154225.98, -286271603.64, -354180680.81, -280735346.39], (24, 10000): [-266600154.6, -277629977.16, -264050733.03, -254567834.32, -306177025.31, -287430282.26, -286093042.91, -282482138.0, -275373499.46, -270484258.4, -288183392.0, -216449055.86, -232702158.09, -290010388.84, -224145421.85], (24, 12500): [-220844219.11, -232119375.94, -219311156.71, -215279404.1, -258861056.79, -242678820.8, -238287919.62, -234594702.08, -229999137.12, -223472905.47, -241847153.19, -179516188.32, -196278629.22, -240791551.13, -187618060.23], (24, 17500): [-170668967.88, -177887764.41, -167916057.59, -162417430.32, -195874373.98, -187505487.17, -180496769.24, -181737548.58, -177084292.2, -170949271.9, -181906204.33, -138578763.15, -149930727.04, -185164406.91, -143591308.21], (24, 25000): [-131545198.24, -135639750.56, -127297022.74, -123486080.12, -152052717.99, -137007353.44, -139425005.08, -139504405.07, -134848642.08, -130994306.74, -141117136.94, -106709795.82, -113963833.69, -141325590.49, -111421638.98]})

    sortedByFrequencies=sorted([t for t,f in id2word_lemmatized.cfs.items()],reverse=True)

    for iter in range(1): 
        maxE=20
        K=20
        datasetsize=25
        random.shuffle(corpus_lemmatized)
        for vocabLimit in [1000]: #,2500,5000,7500,10000,12500,17500,25000]:
            start=datetime.now()
            print('datasetsize = {}k, vocabLimit = {}'.format(datasetsize-1,vocabLimit))

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
            model.vocabLimit=vocabLimit
            model.sortedByFrequencies=sortedByFrequencies

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

                print('elasped {}s bound_h {}'.format(earlystoppingtime,bound_h))

                bounds.append(bound)
                bounds_h.append(bound_h)
                times.append(earlystoppingtime)

                if len(bounds_h)>3:
                    if bounds_h[-1]*(1+10**-5) < bounds_h[-3]*0.5+bounds_h[-2]*0.5:
                        break
            optimalStopping[(datasetsize-1,vocabLimit)].append(times[-1])
            boundsh[(datasetsize-1,vocabLimit)].append(bound_h)

            logging.info('runtimes under dimensions of lambda are: '+str(optimalStopping))
            logging.info('ELBO under dimensions of lambda are:  '+str(eLoopBound))
            logging.info('Please see notebooks for plots')



        logging.info('Time taken = {:.0f} minutes'.format((time.time()-starttime)/60.0))
    

  
