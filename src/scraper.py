from IPython.core.display import HTML
from bs4 import BeautifulSoup, UnicodeDammit
import pandas as pd
import numpy as np
import requests
import StringIO
import json
import pymongo
import multiprocessing as mp
from time import sleep
import random
from requests.adapters import HTTPAdapter
from requests.exceptions import ConnectionError, InvalidURL


def jobs_list(coll , cm, is_indeed):
    title_list = []
    cm = cm.strip().replace(' ', '+')
    
    path = 'http://www.jobs-salary.com/salaries.php?q=&ml=25&lc=&state=&company='+cm+'&title=&sb=date&ps='
    
    
    print 'start'
    
    r = requests.get(path+str(1), headers={'User-agent': 'Mozilla/5.0'}) 
    soup = BeautifulSoup(r.content, from_encoding='UTF-8')
    t = soup.find('div', {'class': 'graybar'})
    pages = t.findAll('a', href=True)[-2].text
    pages = int(pages)
    
    for i in xrange(1, pages+1):
        try:
            r = session_get(path+str(i))
        except ConnectionError:
            time.sleep(30)
            r = session_get(path)
        
        soup = BeautifulSoup(r.content, from_encoding='UTF-8')
        t = soup.find('table', {'class', 'resTb'})
        table = t.findAll('tr')
        
        sleep_list = np.linspace(.1, 1.2, num=10)
        sleep(round(random.choice(sleep_list),4))
        
        
        sal_list = [[x.text for x in row.findAll('td')] for row in table]
    
        for s in sal_list:
            if s:
                title = s[0]
                comp = s[1]
                sal = s[2]
                other = s[3]
                if title not in title_list:
                    #print 'passed'
                    if is_indeed:
                    	title_list.append(title)
                    	search_data_indeed(title, comp, sal, other, coll, cm)
                    else:
                    	title_list.append(title)
                    	search_data_simplyHired(title, comp, sal, other, coll, cm)

def init_db(db_name, db_col):
    # Initiate Mongo client
    client = pymongo.MongoClient()
    # Access database 
    db = client[db_name]
    # Access collection 
    coll = db[db_col]
    # return collection pointer.
    return coll   

# create session per request to avoid overusing a session        
def session_get(path):
    with requests.Session() as s:
        s.mount('http://', HTTPAdapter(max_retries=7))
        r = s.get(path, headers={'User-agent': 'Mozilla/5.0'})
    return r

def search_data_simplyHired(title, company, sal, other,  coll, cm):

    title = title.replace('Tech Yahoo', '').strip()
    title = title.replace('Technical Yahoo', '').strip()
    title = title.replace('Engineer', 'Eng').replace('Eng', 'Engineer').strip()
    title = title.replace('Development', 'Dev').strip()

    
    titl = title
    split_title = titl.split()
    
    
    titl = title.strip().replace(' ', '+')


    if cm == 'yahoo':
        cm = 'yahoo'+'!'
    if cm == 'Amazon Corporate':
        cm = 'Amazon'

    # To make sure only job descriptions for a given company are fetched
    if len(split_title) > 4:
    	path = 'http://www.simplyhired.com/search?q='+cm+'+%22'+titl+'%22+'+split_title[-2]+'%22'+split_title[-1]+'&fcn='+cm+'&ws=10'
    else:
    	path = 'http://www.simplyhired.com/search?q='+cm+'+%22'+titl+'%22+&fcn='+cm+'&ws=10'
        
    

    try:
        r = session_get(path)
    except ConnectionError:
    	print 'error'
        time.sleep(30)
        r = session_get(path)

    
    if r.status_code != 200:
        print 'status code: ', r.status_code
        print 'Number of pages searched:, ', path
        return
    else:
        soup = BeautifulSoup(r.content, from_encoding='UTF-8')
        res = soup.findAll('a', { "class" : "title" }, href= True)
        if not res:
            print 'No more results found'
            return
        else:
            links = map(lambda x: x['href'], res)
            
 
            for link in links:
                get_data_simplyHired('http://www.simplyhired.com' + link, company, title, sal, other, coll)

# to search indead.com and traverse search results 
def search_data_indeed(title, company, sal, other,  coll, cm):

    title = title.replace('Tech Yahoo', '').strip()
    title = title.replace('Technical Yahoo', '').strip()
    title = title.replace('Engineer', 'Eng').replace('Eng', 'Engineer').strip()
    title = title.replace('Development', 'Dev').strip()

    titl = title
    split_title = titl.split()
    titl = title.strip().replace(' ', '+')


    if cm == 'yahoo':
        cm = 'yahoo'+'!'
    if cm == 'Amazon Corporate':
        cm = 'Amazon'
    if cm == 'Hewlett+Packard':
    	cm = 'HP'
    if cm == 'TERADATA OPERATIONS':
    	cm = 'TERADATA'

    
    path = 'http://www.indeed.com/jobs?q=title%3A%28'+titl+'%29+company%3A'+cm+'&filter=0'
    

    try:
        r = session_get(path)
    except ConnectionError:
    	print 'error'
        time.sleep(30)
        r = session_get(path)

    
    if r.status_code != 200:
        print 'status code: ', r.status_code
        print 'Number of pages searched:, ', path
        return
    else:
    	soup = BeautifulSoup(r.content, from_encoding='UTF-8')
    	res = soup.findAll('div', { "class" : "  row  result" })
    	res_last = soup.find('div', { "class" : 'lastRow  row  result' })
    	res.append(res_last)
    	no_res = soup.findAll('div', { "id" : "no_results" })
    	filter(None, res)
        if no_res:
            print 'No more results found'
            #print path
            return
        if res:
        	filter(None, res)
        	links = map(lambda x: x.find('a', {'class':"turnstileLink"}, href = True)['href'], res)
        	for link in links:
        		get_data_indeed('http://www.indeed.com' + link, company, title, sal, other, coll)


def get_data_simplyHired(path, company, title, sal, other, coll):
    
    r = requests.get(path, headers={'User-agent': 'Mozilla/5.0'})
    html = r.content
    soup = BeautifulSoup(r.content, from_encoding='UTF-8')
    job_desc = soup.findAll('div', 'description-full')
    
    sleep_list = np.linspace(.1, 1.2, num=7)
    sleep(round(random.choice(sleep_list),4))
    
    if not job_desc:
        return 
    else:        
        job_desc = job_desc[0].text
        db_insert(company, title, job_desc, sal, other, path, html, coll)

# get data from indeed.com and if no data found return 
def get_data_indeed(path, company, title, sal, other, coll):

    
    print path
    
    try:
    	r = requests.get(path, headers={'User-agent': 'Mozilla/5.0'})
    except InvalidURL:
    	print 'error'
    	return


    html = r.content
    soup = BeautifulSoup(r.content, from_encoding='UTF-8')
    job_desc = soup.text

    
    sleep_list = np.linspace(.1, 1.2, num=7)
    sleep(round(random.choice(sleep_list),4))
    
    if len(job_desc) < 300:
        
        return
    else:
    
        db_insert(company, title, job_desc, sal, other, path, html, coll)



def db_insert(comp, title, job_desc, sal, other, path, html, coll):
    coll.insert({'company':comp, 'title':title, 'desc':job_desc, 'sal':sal, 'other':other,  'url':path, 'html':html})



list_cmp = ['Intel', 'Google', 'Hewlett Packard', 'ebay', 'TERADATA OPERATIONS', 'Twitter', 'IBM', 'yahoo', 'EMC', 'Accenture', 'Amazon Corporate', 'Oracle']
for cm in list_cmp:
    print 'starting '+ cm
    col_MS = init_db('jobs', cm+'q3_salaries')
    jobs_list(col_MS, cm, False)
    print 'Done from'+cm
    sleep(79)


