# Welcome our website for DSC180B A06
Authors:
* Duha Aldebakel
* Yu Cao
* Anthony Limon
* Rui Zhang
## Our project is to explore both Markov Chain Monte Carlo algorithms and variational inference methods for Latent Dirichlet Allocation (LDA). 
This blog is meant to be used by a consumer-facing or general audience, and so we will skip the technical details and give a high-level plain language explanation. For more technical details, please refer to our final paper.
## Latent Dirichlet Allocation (LDA)
Latent Dirichlet Allocation (LDA) is a model that describes how collections of discrete data are generated and tagged. For our purposes, we will focus on text documents about various topics. For example, every Wikipedia page can be considered as a document and each document would be about several topics, such as politics, soccer, history, music, and so on. 
<br>
<br>
We are interested in learning about these topics in an unsupervised manner -- in other words, without humans giving hints or suggestions as to what the topics are. To do get there, we need (1) an underlying model of how documents with their topics are generated, and (2) given this model, fit the parameters with actual live data. LDA would be the solution to the first problem.
<br>
<br>


![](https://github.com/a1limon/DSC180B.visual.io/blob/gh-pages/images/lda_graphical_model.png?raw=true)

Firstly, don’t be alarmed by the many symbols. We can get through this together. What we actually observe is the words on every Wikipedia article. This is represented by the gray circle, with the subscripts denoting which Wikipedia article and word in that article we observe. Everything else in this model is latent variables, that is, not directly observed. Our goal is to infer what these variables are through their collective impact on the observed words. 
<br>
<br>
The rectangles around the circles represent multiplicity. There are N words in every document, and there are D documents in total. There are also K topics.
<br>
<br>
You might also notice that there are two pathways to determine the word in every document. The pathway on the left determines the topic assignment. Is this word about “music” or “sports”, for example? The right pathway determines the word in the vocabulary of the topic. If the topic assignment is “sports”, the vocabulary might consist of words such as “goals”, “soccer”, and “score”, for example.
<br>
<br>
### Topic Assignment

### Topic Vocab 



The intuition behind LDA is the assumption that documents exhibit multiple topics, as opposed to the assumption that documents exhibit a single topic. We can elaborate on this by describing the imaginary generative probabilistic process that we assume our data came from. LDA first assumes that each topic is a distribution over terms in a fixed size vocabulary. LDA then assumes documents are generated as follows:
<br>
* A distribution over topics is chosen
* For each word in a document, a topic from the distribution over topics is chosen
* A word is drawn from a distribution over terms associated with the topic chosen in the previous step.
<br>
In other words, We might choose a topic  from a distribution over topics. Based on this topic $x$ we choose a word from the distribution over terms associated with topic $x$. This is repeated for every word in a document and is how our document is generated. We repeat this for the next document in our collection, and thus a new distribution over topics is chosen and its words are chosen in the same process. It is important to note that the topics across each document remain the same, but the distribution of topics and how much each document exhibits the topics changes. Another important observation to point out is that this model has a bag-of-words assumption, in other words, the order of the words doesn't matter. The generative process isn't meant to retain coherence, but it will generate documents with different subject matter and topics.
