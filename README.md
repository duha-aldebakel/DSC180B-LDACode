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
- SSH to `dsmlp-login.ucsd.edu`
- RUN `launch-scipy-ml.sh -i <image-fullname> -P Always` . The `-P Always` flag will force the docker host to sync, as it pulls the latest version of the image manifest. Note: a docker image name follows the format `<user>/<image>:<tag>`. The `:<tag>` part will be assumed to be `:latest` if you don't supply it to the launch script. Use tags like `v1` or `test` in the build step to have control over different versions of the same docker image.
- In our case, the command would be "`launch-scipy-ml.sh -i daldebak/dsc180b:latest -P Always`"
  
  

- `git clone https://github.com/duha-aldebakel/DSC180B-LDACode.git`
- `cd DSC180B-LDACode`
- Option 2: 
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

# Build and deploy

At this point you've made changes to the repository on a separate branch that serves as your local working copy. After verifying it works locally you can create a pull request which will in turn run a workflow that builds an image from the `Dockerfile`. The workflow is set to run on a pull request which verifies not only that there are no merge conflicts, but that the image is successfully built with the changes made. Though, the expected changes should really only pertain to the `requirements.txt`.

