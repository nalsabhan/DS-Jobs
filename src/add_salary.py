from bs4 import BeautifulSoup, UnicodeDammit
import pandas as pd
import numpy as np
import requests
import StringIO
import json
import pymongo
from time import sleep
import random
from requests.adapters import HTTPAdapter
from requests.exceptions import ConnectionError, InvalidURL

'''
To get all salaries from jobs-salary.com for the retrieved job titles and their descriptions
'''


def init_db(db_name, db_col):
    # Initiate Mongo client
    client = pymongo.MongoClient()

    # Access database 
    db = client[db_name]
    
    # Access collection 
    coll = db[db_col]

    # return collection pointer.
    return coll

def insert_sal(comp, title, sal, oth, coll):
    coll.insert({'company':comp, 'title':title, 'salary':sal, 'other':oth})

# create session per request to avoid overusing a session
def session_get(path):
    with requests.Session() as s:
        s.mount('http://', HTTPAdapter(max_retries=7))
        r = s.get(path, headers={'User-agent': 'Mozilla/5.0'})
    return r

# main function that calls get salary per company
def main_func(company, col_name_base):

    coll_out = init_db('jobs', company+col_name_base)
    coll_in = init_db('jobs', '2salary_'+company)
    ttl_list = coll_out.distinct("title")

    for ttl in ttl_list:
        get_salaries(company, ttl, coll_in)

def get_salaries(company, title, coll_in):
    if company == MS:
        company =
    cm = company.strip().replace(' ', '+')
    ttl = title.strip().replace(' ', '+')

    path = 'http://www.jobs-salary.com/salaries.php?q=&ml=25&lc=&state=&company='+cm+'&title='+ttl+'&sb=date&ps='

    r = requests.get(path, headers={'User-agent': 'Mozilla/5.0'}) 
    soup = BeautifulSoup(r.content, from_encoding='UTF-8')
    gray_bar = soup.find('div', {'class': 'graybar'})
    print 'start'

 
    
    res = soup.find('table', {'class', 'resTb'})

    cnt = 0
    while res == None and cnt < 7:
        print title
        r = session_get(path)
        soup = BeautifulSoup(r.content, from_encoding='UTF-8')
        res = soup.find('table', {'class', 'resTb'})
        if cnt == 0:
            ti = ttl.replace('Dev', 'Development')
        elif cnt == 1:
            ti = ttl.replace('Engineer', 'Eng', 1)
        elif cnt == 2:
            ti = ttl.replace('Engineer','Eng', 2)
        elif company == 'yahoo' and cnt == 3:
            ti = 'Tech Yahoo '+ttl
        elif company == 'yahoo' and cnt == 4:
            ti = ttl+' Tech Yahoo'
        elif company == 'yahoo' and cnt == 5:
            ti = 'Technical Yahoo '+ttl
        elif company == 'yahoo' and cnt == 6:
            ti = ttl+' Technical Yahoo'
        elif cnt == 7:
            ti = ttl.replace('Dev', 'Development')
        else:
            print title
        path = 'http://www.jobs-salary.com/salaries.php?q=&ml=25&lc=&state=&company='+cm+'&title='+ti+'&sb=date&ps='
        cnt+=1


    if res!=None:
        if gray_bar:
            pages = gray_bar.findAll('a', href=True)[-2].text
            pages = int(pages)
        else:
            pages = 1
        
        for i in xrange(1, pages+1):
            if i > 1:
                try:
                    r = session_get(path+str(i))
                except ConnectionError:
                    time.sleep(30)
                    r = session_get(path)
            
            #r = requests.get(path+str(i), headers={'User-agent': 'Mozilla/5.0'})
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
                    insert_sal(comp, title, sal, other, coll_in)
    else:
        with open('out2.txt', 'a') as f:
            f.write(company+': '+title+'\n')
    



#comp_list = ['MS', 'Intel', 'Google', 'Hewlett Packard', 'ebay', 'Twitter', 'IBM', 'EMC', 'Accenture', 'Amazon Corporate', 'Oracle', 'yahoo']
comp_list = ['MS']
for comp in comp_list:
    main_func(comp, 'q3_salaries')











