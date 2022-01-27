# wikidownloader.py: 
# Download random articles from Wikipedia 
#  and save them in compressed python3 pickle format.
#
# Duha Aldebakel
# Reference: https://www.stats.ox.ac.uk/~teh/sgrld.html
# Stochastic Gradient Riemannian Langevin Dynamics for Latent Dirichlet Allocation
# Sam Patterson and Yee Whye Teh

import sys
import re, string, time, threading
import pickle
import bz2
import requests
import random

#20220124 Update
#These libraries are used for tokenizing, removing stop words and stemming.
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
import pandas as pd

tokenizer = RegexpTokenizer(r'\w+')

# create English stop words list
en_stop = get_stop_words('en')

# Create p_stemmer of class PorterStemmer
p_stemmer = PorterStemmer()

    

def get_random_wikipedia_article():
    url='http://en.wikipedia.org/wiki/Special:Random'
    url2='http://en.wikipedia.org/w/index.php?title=Special:Export/{}&action=submit'
    headers={
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_{}) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.47 Safari/537.36'.format(random.randint(0,9))
        }
    while True:
        articletitle = None
        alltext = None
        line = None
        try:
            resp = requests.get(url, timeout=5)
            if resp.status_code!=200:
                print('{} Error (Probably reached rate limit)'.format(resp.status_code))
                time.sleep(10) #Possible rate limiting from wiki
                continue #try again
            line=resp.text
            result = re.search('title="Edit this page" href="/w/index.php\?title=(.*)\&amp;action=edit"/\>', line)
            if (result):
                articletitle = result.group(1)
            else:
                print('No articletitle')
                continue #try again
            if not articletitle :
                print('No articletitle')
                continue #try again
                
            resp = requests.get(url2.format(articletitle), timeout=5)
            if resp.status_code!=200:
                print('{} Error (Probably reached rate limit)'.format(resp.status_code))
                time.sleep(10) #Possible rate limiting from wiki
                continue #try again
            alltext = resp.text
        except requests.exceptions.Timeout:
            print('I waited far too long')
            continue
        except:
            continue #try again
        try:
            alltext = re.search(r'<text.*?>(.*)</text', alltext, flags=re.DOTALL).group(1)
            for white in ['\n','\r','&amp;nbsp;',"&nbsp"]:
                alltext=alltext.replace(white,' ')
                
            alltext = re.sub(r'\{\{.*?\}\}', r'', alltext)
            alltext = re.sub(r'\[\[Category:.*', '', alltext)
            alltext = re.sub(r'==\s*[Ss]ource\s*==.*', '', alltext)
            
            
            alltext = re.sub(r'==\s*[Rr]eferences\s*==.*', '', alltext)
            alltext = re.sub(r'==\s*[Ee]xternal [Ll]inks\s*==.*', '', alltext)
            alltext = re.sub(r'==\s*[Ee]xternal [Ll]inks and [Rr]eferences==\s*', '', alltext)
            alltext = re.sub(r'==\s*[Ss]ee [Aa]lso\s*==.*', '', alltext)
            alltext = re.sub(r'http://[^\s]*', '', alltext)
            alltext = re.sub(r'\[\[Image:.*?\]\]', '', alltext)
            alltext = re.sub(r'Image:.*?\|', '', alltext)
            alltext = re.sub(r'\[\[.*?\|*([^\|]*?)\]\]', r'\1', alltext)           
            
            #remove wiki timeline objects
            #Example https://en.wikipedia.org/w/index.php?title=Special:Export/Mornago&action=submit&action=submit
            #https://en.wikipedia.org/wiki/Mornago
            #...ImageSize  = width:455 height:303 PlotArea   = left:50 bottom:50 top:30 right:30 DateFormat = x.y Period     = from:0 till:5000 TimeAxis   = orientation:vertical AlignBars  = justify ScaleMajor = gridcolor:darkgrey increment:1000 start:0 ScaleMinor = gridcolor:lightgrey increment:200 start:0 BackgroundColors = canvas:sfondo  BarData=   bar:1861 text:1861
            alltext = re.sub(r'&lt;timeline&gt;.*&lt;/timeline&gt;', '', alltext)
            

            #old code only removed http and not https
            #this caused http to show up as a topic, for example: 
            #For https://en.wikipedia.org/wiki/Geoffrey_Cantor
            #   0.030*"amp" + 0.019*"http" + 0.019*"univers"
            #note http is not actually in the text
            alltext = re.sub(r'https://[^\s]*', '', alltext) 
            
            #Remove &amp;
            alltext = re.sub(r'&amp;', ' ', alltext) 
            
            
            
            #some topics show up which appears to be wiki table related
            #This code here removes all wikitables
            #   '0.163*"align" + 0.130*"style" + 0.116*"center" + 0.063*"background" + 0.055*"text" + 0.034*"bgcolor" + 0.034*"width" + 0.025*"right"'
            #Example 
            #https://en.wikipedia.org/w/index.php?title=Special:Export/UD_Talavera&action=submit&action=submit
            alltext=re.sub(r'{\|.*\|}','',alltext)
            
            
            #we obtained the following formula for a topic but lots of wiki related tags
            #'0.089*"align" + 0.062*"right" + 0.052*"bgcolor" + 0.049*"id" 
            #+ 0.025*"km" + 0.022*"center" + 0.021*"text" + 0.021*"mount" + 
            #0.021*"bar" + 0.021*"style"'

            #remove example: align=right
            #'0.089*"align" + 0.062*"right"
            alltext = re.sub(r'align=[^\s]*', '', alltext)
            
            #remove example: -id=000 bgcolor=#E9E9E9
            #remove example: data-sort-value="0.72" 
            alltext = re.sub(r'id=[^\s]*', '', alltext)
            alltext = re.sub(r'bgcolor=[^\s]*', '', alltext)
            alltext = re.sub(r'fontsize:[^\s]*', '', alltext)
            alltext = re.sub(r'data-sort-value=[^\s]*', '', alltext)
            
            #remove example: fontsize:XS
            alltext = re.sub(r'bar:[^\s]* from: [^\s]* till:[^\s]*','',alltext)

            alltext = re.sub(r'\&lt;.*?&gt;', '', alltext)
            

        except:
            print('Parse error')
            continue #try again

        break

    return(alltext, articletitle)

class WikiThread(threading.Thread):
    articles = []
    articlenames = []
    tokens = []
    stemmed_tokens = []
    lock = threading.Lock()
    numdownload = 50000
    StopEvent = 0
    numrunning = 0
      
    def __init__(self,args):
        threading.Thread.__init__(self)
        self.StopEvent = args    
        self.id=WikiThread.numrunning
        WikiThread.numrunning+=1

    def run(self):
        while True:
            (article, articlename) = get_random_wikipedia_article()

            # clean and tokenize document string
            raw = article.lower()
            ts = tokenizer.tokenize(raw)
            
            # remove stop words, numerics and single chars from tokens
            stopped_ts = [i for i in ts if not i in en_stop and 
                          not i.isnumeric() and len(i) > 1]
            
            # stem tokens
            stemmed_ts = [p_stemmer.stem(i) for i in stopped_ts]
    
            with WikiThread.lock:
                WikiThread.articles.append(article)
                WikiThread.articlenames.append(articlename)
                WikiThread.tokens.append(ts)
                WikiThread.stemmed_tokens.append(stemmed_ts)
                
                if WikiThread.numdownload<len(WikiThread.articlenames):
                    break
            if (self.StopEvent.wait(0)):
                break;
            
        print ("Thread {} terminated".format(self.id))



if __name__ == '__main__':
#while True: #Change to this line to collect a lot of data =)
    #Load old data
    print('Load old data')
    try:
        #with open('data.pickle', 'rb') as f:
        with bz2.BZ2File('datapickle.bz2', 'rb') as f:  #Use datacompression BZ2
            olddata = pickle.load(f)
    except:
        olddata=([],[],[],[])
    print('We have {} documents'.format(len(olddata[0])))
    maxthreads = 4 #Higher than 4 seems to get into trottling via 429 errors

    wtlist = []    
    for i in range(maxthreads):
        Stop = threading.Event()
        thisthread=WikiThread(Stop)
        thisthread.start()
        wtlist.append((Stop,thisthread))
    while WikiThread.numdownload>len(WikiThread.articlenames):
        time.sleep(10)
        print('{}/{} downloaded. Last title was {}'.format(len(WikiThread.articlenames),WikiThread.numdownload,WikiThread.articlenames[-1]))
    for s,t in wtlist:
        # ask(signal) the child thread to stop
        s.set() 
        t.join(5)
    
    newdata=(WikiThread.articles, WikiThread.articlenames)
    
    #Save new data
    print('Save new data')
    #with open('data.pickle', 'wb') as f:
    with bz2.BZ2File('datapickle.bz2', 'wb') as f: #Use datacompression BZ2
        pickle.dump((olddata[0]+WikiThread.articles,
                     olddata[1]+WikiThread.articlenames,
                     #olddata[2]+WikiThread.tokens,
                     #olddata[3]+WikiThread.stemmed_tokens,
                     ), f)
    print('All done')
    WikiThread.articles=[]
    WikiThread.articlenames=[]
    
'''
Example of one document
pd.DataFrame({'text':data[0],'title':data[1],'tokens':data[2],'stemmed':data[3]}).loc[0]
Out[31]: 
text       'Alcyone Cone' () is an extinct volcanic cone near the center of The Pleiades, at the west s...
title                                                                                             Alcyone_Cone
tokens     [alcyone, cone, is, an, extinct, volcanic, cone, near, the, center, of, the, pleiades, at, the, ...
stemmed    [alcyon, cone, extinct, volcan, cone, near, center, pleiad, west, side, head, marin, glacier, vi...
            
            
'''