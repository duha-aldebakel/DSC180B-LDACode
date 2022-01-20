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
            alltext = re.sub(r'\&lt;.*?&gt;', '', alltext)
        except:
            print('Parse error')
            continue #try again

        break

    return(alltext, articletitle)

class WikiThread(threading.Thread):
    articles = []
    articlenames = []
    lock = threading.Lock()
    numdownload = 1000
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
            with WikiThread.lock:
                WikiThread.articles.append(article)
                WikiThread.articlenames.append(articlename)
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
        olddata=([],[])
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
    #    pickle.dump(anything,f)
    with bz2.BZ2File('datapickle.bz2', 'wb') as f: #Use datacompression BZ2
        pickle.dump((olddata[0]+newdata[0],olddata[1]+newdata[1]), f)
    print('All done')
    WikiThread.articles=[]
    WikiThread.articlenames=[]
    
