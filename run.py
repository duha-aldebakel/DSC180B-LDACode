import pandas as pd
import logging
logger = logging.getLogger(__name__)
import logging.handlers
import sys
import yaml
from yaml.error import YAMLError
from src.etl import load_data
from src.lda_gensim import run_lda_gensim

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

   else:
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
      return

if __name__ == "__main__":
    targets = sys.argv[1:]
    main(targets)
