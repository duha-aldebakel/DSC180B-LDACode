# Overview & Setup

This repository serves as the central user code repository for Group A06's DSC180B Capstone project.

Title:
Exploration of Variational Inference and Monte Carlo Markov Chain Models for Latent Dirichlet Allocation of Wikipedia Corpus
Abstract:
Topic modeling allows us to fulfill algorithmic needs to organize, understand, and annotate documents according to the discovered structure. Given the vast troves of data and the lack of specialized skillsets, it is helpful to extract topics in an unsupervised manner using Latent Dirichlet Allocation (LDA). LDA is a generative probabilistic topic model for discrete data, but unfortunately, solving for the posterior distribution of LDA is intractable, given the numerous latent variables that have cross dependencies. It is widely acknowledged that inference methods such Markov Chain Monte Carlo and Variational Inference are a good way forward to achieve suitable approximate solutions for LDA. In this report, we will explore both these methods to solve the LDA problem on the Wikipedia corpus. We find that better performance can be achieved via preprocessing the data to filter only certain parts-of-speech via lemmatization, and also exclude extremely rare or common words. We improved on the Expectations-Maximization (EM) Algorithm used for variational inference by limiting the number of iterations in the E step even if sub-optimal. This leads to benefit of faster runtimes and better convergences due to fewer iterations and avoidance of local minima. Finally, we explore early stopping runtimes on under-parameterized LDA models to infer the true dimensionality of the Wikipedia vocabulary to solve for topics. While the English language has around a million words, our findings are that it only takes around fifteen thousand words to infer around twenty major topics in the dataset.

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
### For local development:
- Make sure, preferably, you have python3.7+ installed and assuming you have configured the `PATH` and `PATHEXT` variables upon installation:
- `python3.7 -m venv env`
- `source env/bin/activate`
- `pip install -r requirements_pip.txt`

To interact with jupyter notebooks (make sure virtual environment is activated and requirements_pip.txt are installed):
- `cd DSC180B-LDACode`
- `ipython kernel install --user --name=env` (assuming you named your virtual environment `env`)
-  `jupyter notebook`
- When notebook server is running: Navigate to `Kernel` > `Change kernel` > select `env`
  
## 2) Getting the repository from Github
- "`git clone https://github.com/duha-aldebakel/DSC180B-LDACode.git`"
- "`cd DSC180B-LDACode`"
- "`python run.py test-gensim`" to run on test data
- "`python run.py`" to run on production data (run time ~16-17min)
  
