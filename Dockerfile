# 1) choose base container
FROM ucsdets/datascience-notebook:2021.2-stable

# 2) change to root to install packages
USER root
#RUN pip install --no-cache-dir geopandas babypandas 
RUN pip install --no-cache-dir pymc3 pandas
COPY requirements_pip.txt requirements_pip.txt
RUN pip install --no-cache-dir -r requirements_pip.txt

# 3) Download language library data
RUN python -m spacy download en_core_web_sm
RUN python -m spacy download en_core_web_md
RUN python -m nltk.downloader stopwords

#rename this file as "Dockerfile" then run:-
#docker build -t duha-aldebakel/dsc180b .

#3afa5723-df1c-4979-a0dc-2dd3d18abb30
