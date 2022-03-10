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

To interact with jupyter notebooks (make sure virtual environment is activated and requirements.txt are installed):
- `cd DSC180B-LDACode`
- `ipython kernel install --user --name=env` (assuming you named your virtual environment `env`)
-  `jupyter notebook`
- When notebook server is running: Navigate to `Kernel` > `Change kernel` > select `env`
  
## 2) Getting the repository from Github
- "`git clone https://github.com/duha-aldebakel/DSC180B-LDACode.git`"
- "`cd DSC180B-LDACode`"
- "`python run.py test-gensim`" to run on test data
- "`python run.py`" to run on production data (run time ~16-17min)
  
