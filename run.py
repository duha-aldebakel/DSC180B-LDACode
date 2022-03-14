import pandas as pd
import logging
logger = logging.getLogger(__name__)
import logging.handlers
import sys
import yaml
from yaml.error import YAMLError
from src.etl import load_data
from src.lda_gensim import run_lda_gensim
from src.lda import run_cgs
from src.onlineldavb import *

def main(targets):
   logging.basicConfig(
         filename='logs/app.log', 
         filemode='w',
         format='%(asctime)s - %(name)s %(levelname)s %(message)s', 
         datefmt='%d-%b-%y %H:%M:%S',
         level=logging.DEBUG
   )
   logging.info('Number of arguments: {} arguments.'.format(len(sys.argv)))
   logging.info( 'Argument List: {}'.format(str(sys.argv)))
   if 'test-gensim' in targets:
      with open("config/test_gensim_config.yaml", "r") as f:
         try:
            test_config = yaml.safe_load(f)
         except YAMLError as exc:
            print(exc)
     
      logging.info('App started in test mode. Using test data (generated with just 100 articles. See  wikidownloader.py)')
      logging.info('\n\n# Loading data pickle file '+ test_config['data_dir'] + test_config['file_name'])
      data = load_data(**test_config)
      lda_gensim_params = test_config['lda_gensim_params']
      df = pd.DataFrame({'text':data[0],'title':data[1]})
      #remove zero length articles
      articlelen=df.text.apply(len)
      df=df[articlelen>10]
      run_lda_gensim(corpus=df, **lda_gensim_params)

   elif 'gensim' in targets:
      with open("config/gensim_config.yaml", "r") as f:
         try:
            config = yaml.safe_load(f)
         except YAMLError as exc:
            print(exc)
      
      
      logging.info('App started in prod mode. Using approximately 50,000 articles.')
      logging.info('\n\n# Loading data pickle file '+ config['data_dir'] + config['file_name'])
      data = load_data(**config)
      lda_gensim_params = config['lda_gensim_params']
      df = pd.DataFrame({'text':data[0],'title':data[1]})
      #remove zero length articles
      articlelen=df.text.apply(len)
      df=df[articlelen>10]
      run_lda_gensim(corpus=df, **lda_gensim_params)
   
   elif 'test-lda-cgs' in targets:
      with open("config/test_lda_config.yaml", "r") as f:
         try:
            config = yaml.safe_load(f)
         except YAMLError as exc:
            print(exc)

      data = load_data(**config)
      df = pd.DataFrame({'text':data[0],'title':data[1]})
      #remove low length articles
      articlelen=df.text.apply(len)
      df=df[articlelen>10]
      data = df.text.values
      data = data[:500]
      lda_cgs_params = config['lda_params']
      run_cgs(data, **lda_cgs_params)

   elif 'lda-cgs' in targets:
      with open("config/lda_config.yaml", "r") as f:
         try:
            config = yaml.safe_load(f)
         except YAMLError as exc:
            print(exc)

      data = load_data(**config)
      df = pd.DataFrame({'text':data[0],'title':data[1]})
      #remove low length articles
      articlelen=df.text.apply(len)
      df=df[articlelen>10]
      data = df.text.values
      lda_cgs_params = config['lda_params']
      run_cgs(data, **lda_cgs_params)
   else:
      with open("config/gensim_config.yaml", "r") as f:
         try:
            config = yaml.safe_load(f)
         except YAMLError as exc:
            print(exc)
      logging.info('App started in prod mode. Using approximately 50,000 articles.')
      logging.info('Defaulting to python run.py onlineldavb (Blei/Hoffman code modified)')
      logging.info('Please consider other command line options as well.')
      logging.info('\n\n# Loading data pickle file '+ config['data_dir'] + config['file_name'])
      data = load_data(**config)
      lda_gensim_params = config['lda_gensim_params']
      df = pd.DataFrame({'text':data[0],'title':data[1]})
      #remove zero length articles
      articlelen=df.text.apply(len)
      df=df[articlelen>10]
      run_onlineldavb(corpus=df, **lda_gensim_params)

if __name__ == "__main__":
    targets = sys.argv[1:]
    main(targets)
