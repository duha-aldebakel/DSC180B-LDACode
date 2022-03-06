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

The intuition behind LDA is the assumption that documents exhibit multiple topics, as opposed to the assumption that documents exhibit a single topic. We can elaborate on this by describing the imaginary generative probabilistic process that we assume our data came from. LDA first assumes that each topic is a distribution over terms in a fixed size vocabulary. LDA then assumes documents are generated as follows:
<br>
* A distribution over topics is chosen
* For each word in a document, a topic from the distribution over topics is chosen
* A word is drawn from a distribution over terms associated with the topic chosen in the previous step.
<br>
