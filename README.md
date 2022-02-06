# Overview & Setup

This repository serves as the central user code repository for Group A06's DSC180B Capstone project.

To get it up and running:
- `git clone https://github.com/duha-aldebakel/DSC180B-LDACode.git`
- `cd DSC180B-LDACode`
- Option 1:
- Build from Dockerfile using the command 'docker build -t daldebak/dsc180b .'
- Run the docker using 'docker run -it --rm daldebak/dsc180b bash '
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

