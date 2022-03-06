# Overview & Setup

This repository serves as the central user code repository for Group A06's DSC180B Capstone project.

To get it up and running:
## 1) Set up python environment
### Option 1: (Easiest) Pulling our Docker from Dockerhub
- Just run "`docker run -it --rm daldebak/dsc180b bash`"
- If the docker is not on your machine locally, the command should pull it from docker hub automatically. Here is how to force the action, "`docker pull daldebak/dsc180b`"
- Run the docker using 'docker run -it --rm daldebak/dsc180b bash '
### Option 2: Rebuilding a Docker
- Build from Dockerfile using the docker CLI command
- Type "`docker build -t <image-fullname> .`" and hit \<Enter\>, notice the "period/dot" at the end of the command, which denotes the current directory. Docker will then build the image in the current directory's context. The resulting image will be labeled `<image-fullname>`. Monitor the build process for errors.
- For example, a command could be 'docker build -t daldebak/dsc180b .'
- Run the docker using 'docker run -it --rm daldebak/dsc180b bash ', or replace with your own <image-fullname>
### Option 3: If using DSMLP
- SSH to `dsmlp-login.ucsd.edu`. (Note if working outside the school, you would need to first connect via VPN)
- Run "`launch.sh -i daldebak/dsc180b:latest`"
### Option 4: Running from your own python environment (Hardest)
- Make sure, preferably, you have python3.7+ installed and assuming you have configured the `PATH` and `PATHEXT` variables upon installation:
- `python3.7 -m venv env`
- `source env/bin/activate`
- `pip install -r requirements.txt`
To interact with jupyter notebooks (make sure virtual env is activated and requirements.txt are installed):
- `cd DSC180B-LDACode`
-  Download spacy data: 
-  `python3 -m spacy download en_core_web_md`
-  `python3 -m spacy download en_core_web_sm`
-  Additional downloads...
-  `jupyter notebook`
  
## 2) Getting the repository from Github
- "`git clone https://github.com/duha-aldebakel/DSC180B-LDACode.git`"
- "`cd DSC180B-LDACode`"
- "`python run.py test`" to run on test data
- "`python run.py`" to run on production data
  

The intuition behind LDA is the assumption that documents exhibit multiple topics, as opposed to the assumption that documents exhibit a single topic. We can elaborate on this by describing the imaginary generative probabilistic process that we assume our data came from. LDA first assumes that each topic is a distribution over terms in a fixed size vocabulary. LDA then assumes documents are generated as follows:
\begin{enumerate}
\item A distribution over topics is chosen
\item For each word in a document, a topic from the distribution over topics is chosen
\item A word is drawn from a distribution over terms associated with the topic chosen in the previous step.
\end{enumerate}
In other words, We might choose a topic $x$  from a distribution over topics. Based on this topic $x$ we choose a word from the distribution over terms associated with topic $x$. This is repeated for every word in a document and is how our document is generated. We repeat this for the next document in our collection, and thus a new distribution over topics is chosen and its words are chosen in the same process. It is important to note that the topics across each document remain the same, but the distribution of topics and how much each document exhibits the topics changes. Another important observation to point out is that this model has a bag-of-words assumption, in other words, the order of the words doesn't matter. The generative process isn't meant to retain coherence, but it will generate documents with different subject matter and topics.

Now that we have explained the generative process, we can reiterate the statistical problem that is we cannot actually observe the hidden structure, we only assume it exists. Thus, we want to solve this problem by inferring all of the values of the hidden variables: the topic proportions associated with each document, the topics associated with each word, and the distribution over terms that forms the documents in a collection.

LDA as a graphical model allows us to describe the generative process as well as define a factorization of the joint probability of all of the hidden and observed random variables. It can help us infer the hidden variables given the observations by writing down the algorithms that can solve this problem. The joint probability defines a posterior in which we want to infer from a collection of documents, the topic assignments for each word $z_{d,n}$, the topic proportions for each document $\theta_{d}$, and the topic distributions for each corpus $\beta_{k}$. We can then use those posterior inferences to perform varying tasks. In summary, hidden structure is uncovered by computing the posterior, and then the uncovered structure can be used to perform tasks. This is done by using all the hidden variables that we assume existed in the collections of data, and discovered through the posterior distribution.
