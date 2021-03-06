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

![Latent Dirichlet Allocation](https://github.com/a1limon/DSC180B.visual.io/blob/gh-pages/images/lda_graphical_model.png?raw=true)

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
Starting from the far left, alpha is a hyperparameter that determines how concentrated the topics are. Would a random Wikipedia document discuss many topics, or would it focus on a small number of topics? Typically, it does not make sense that a document would talk about everything under the sun, and so we will expect some concentration of topics as a good author is wise to focus his or her energies.
<br>
<br>
Next, based on alpha and the Dirichlet distribution, each document receives its own theta value which determines the overall proportion of topics in the document. (We can see now that we have D thetas as we enter the rectangle marked D) Perhaps one theta value might be 50% “health” and 50% “food”, if the article is about the healing ability of good cooking! 
<br>
<br>
Next, based on theta and the multinomial distribution, each word in every document receives its own topic assignment, making D x N assignments in total. The model is a bag of words, so the ordering does not matter. Based on our previous example of 50% “health” and 50% “food”, each word would be flipping a coin and getting exactly only one of those assignments.

### Topic Vocab 
From the above, we have topic assignments for every word in every document. How is the actual word chosen? Starting from the far right, we have another hyperparameter of the Dirichlet distribution, to determine the concentration of the vocabulary of each topic. Since there are K topics, we draw from the Dirichlet distribution K times to get a discrete probability distribution of the vocabulary of each topic. For example, if the topic is “food”, the probabilities of words such as “sushi”, “rice”, “egg” would be higher than other topics.
<br>
<br>
Putting it together, we will condition on the chosen topic for the word from the topic assignment on the left pathway. This looks up the appropriate vocabulary for that topic. Using another multinomial distribution, we basically roll a weighted die customized for that topic to choose a random word assignment. This results in actual words that we observe.
<br>
<br>
This process is called Latent Dirichlet Allocation. The term “latent” refers to the fact that many of the parameters cannot be observed directly. The term “Dirichlet Allocation” refers to the manner in which we assign discrete probabilities on topic proportions.
# Bayesian Inference
Now that we have a generative model on how documents are generated from topics, the next step is to fit the model on live data. This process is called Bayesian Inference, which maximizes the likelihood of the latent parameters to match what we have seen from live data. 
<br>
<br>
The issue is that solving this problem is intractable. This means it is difficult or impossible to solve analytically. The problem arises from the dependence of variables from the two pathways of “Topic Assignment” and “Topic Vocab”, where solving each pathway would require knowledge of the other.
<br>
<br>
Instead of trying to find the exact solution, it would thus be better to find a good approximate solution. Along this line, Mean-field variational inference and Gibbs Sampling are two popular ways to solve the inference problem for the latent parameters. 
## Mean-field variational inference
### Variational inference
The technical term for the correct distribution given the observed data is the posterior, denoted by P, which is often not a “nice” distribution to deal with mathematically. The posterior can be any distribution at all and doesn’t have to be a well-known distribution that is mathematically easy to manipulate. The solution is to choose a distribution Q from a user-defined variational family that approximates P. (We choose Q within the variational family that is “closest” to P, using measures such as KL divergences.) This process of choosing a variational family, and then finding the closest distribution Q to P within that variational family is known as variational inference. 
<br>
<br>
![Variational Inference](https://miro.medium.com/max/1002/1*YVFAbC7DgfAj94-0TRt8IQ.png)
<br>
<br>
The graph above illustrates this where the posterior P which is conditioned on the observed data X has an unknown blue distribution. We use a Gaussian variational family, and vary the parameters which are in this case the mean and standard deviation to find the best Q to match the posterior. It would not be a perfect match, and thus this is an approximate solution. Obviously, the choice of the variational family would also determine how close Q will ultimately be to P.
<br>
<br>
## Mean-field variational inference
Mean-field variational inference is a special case of variational inference where we further assume that the variational family can be fully factored. In other words, the latent parameters are all independent of each other. Recall that our main problem was due to the dependence of the variables on the left and right pathways to the observed data, with conditional probabilities involved in the definition. If we instead choose to break this and assume independence, the variational family would not have to deal with conditional probabilities but instead, be simply the product of all the multinomial and Dirichlet distributions. Explicitly:
<br>
<br>
![]( https://github.com/duha-aldebakel/DSC180B-LDACode/blob/master/images/MFVI.PNG?raw=true)
<br>
<br>
The template for Mean-field variational inference thus looks like the image below. The arrows are no longer linked since the variables are independent. In addition, you would notice variational parameters phi and gamma. (And while not shown, there is also lambda as the variational parameter for beta.) These are variational parameters that are used to tweak the Q distribution to be as close to P as possible.
<br>
<br>
![Mean-field Variation Inference](https://camo.githubusercontent.com/be972716d5117d7d769095628c721909c5f4f90ece185c99054e936beee1435d/68747470733a2f2f6769746875622e636f6d2f647568612d616c646562616b656c2f445343313830422d4c4441436f64652f626c6f622f6d61737465722f696d616765732f4c44415f4d65616e6669656c642e504e473f7261773d74727565)
<br>
<br>
We can now hill-climb our way to find the best Q using the Expectation-Maximization (EM) Algorithm. This involves the E step that matches theta and z from the left pathway for topic proportions, with the assumption that every other latent parameter is already correct when independently optimizing a single latent parameter. (This is done by optimizing the variational parameters of phi and gamma for z and theta respectively.) 
<br>
<br>
Then, the M step matches the beta from the right pathway for topic vocab, while again assuming that the latent parameters from the E step are correct due to independence. (This is done by optimizing the variational parameters of lambda for beta.) 
<br>
<br>
The pseudo-code of the EM algorithm is shown below.
![EM Algo](https://camo.githubusercontent.com/22fce9cd102f74289c87d8834c04081b9a909f1861520e0da6f8c3a5eb2d5cfe/68747470733a2f2f6769746875622e636f6d2f647568612d616c646562616b656c2f445343313830422d4c4441436f64652f626c6f622f6d61737465722f696d616765732f56425f4c44415f416c676f312e504e473f7261773d74727565)
## Gibbs Sampling
Another way to solve the Bayesian inference problem of LDA is via Gibbs Sampling. 
<br>
<br>
Sampling is a method to use random numbers to draw samples in an attempt to find the probability distribution of the posterior, for example, to produce a histogram of the distribution. 
<br>
<br>
Gibbs Sampling is a specialized method of sampling that falls under the category of Markov chain Monte Carlo (MCMC) algorithms. This technique involves choosing the next sample based on taking an informed step from the last sample in the chain. In this matter, more time can be spent sampling the distribution which the highest contribution to the probability distribution, hence more efficiently solving the problem.
# Final output
We spent a lot of time discussing how to infer the latent parameters of LDA. The final output that is actually of most interest is the topic vocab or beta. We ran our experiment on documents from Wikipedia and here are the vocabularies of the first two topics.
![](https://github.com/duha-aldebakel/DSC180B-LDACode/blob/master/images/Topics1.PNG?raw=true)
<br>
<br>
The first topic above appears to be about storybooks, as you can see from the relative frequencies of “story”, “book” and “character” which appear with higher probability than other documents on different topics.
<br>
<br>
![](https://github.com/duha-aldebakel/DSC180B-LDACode/blob/master/images/Topics2.PNG?raw=true)
<br>
<br>
The second topic above appears to be about the geography of a village or town.
<br>
<br>




