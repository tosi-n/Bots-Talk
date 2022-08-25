import requests
import pandas as pd
from bs4 import BeautifulSoup
from urllib.request import urlopen


# Make a request
# page = requests.get(
#     "https://www.brides.com/how-social-media-affects-relationships-5105350")#https://www.psychologytoday.com/us/blog/close-encounters/201410/6-myths-about-men-women-and-relationships https://www.gottman.com/about/research/couples/
# soup = BeautifulSoup(page.content, 'html.parser')# html.parser html5lib

sources = { 'body':'https://www.healthline.com/health/types-of-relationships#a-c', 'article':'https://www.brides.com/how-social-media-affects-relationships-5105350', 'p':'https://www.gottman.com/about/research/couples/', 'body':'https://www.psychreg.org/define-relationship-dynamics/', 'body':'https://www.universalclass.com/articles/psychology/healthy-relationship-dynamics.htm' }
# Make a request

_id = []
url = []
body = []


for (index, (k,v)) in enumerate(sources.items()):
    print('='*20)
    page = urlopen(v)# 
    soup = BeautifulSoup(page.read(), 'html.parser')# html.parser html5lib


    table = soup.findAll([k])

    for row in table:
        _id.append(index)
        print(index)
        url.append(v)
        print(v)
        body.append(row.get_text())
        print(row.get_text())

    print('='*20)

print(len(_id))
print(len(url))
print(len(body))

df = pd.DataFrame([_id, url, body]).transpose()
df.columns = ['_id', 'url', 'body']

df.to_csv('./data/evidence_ep1.csv', sep='\t', encoding='utf-8', index=False, header=True)

# [['_id', 'url', 'body']]


# page = urlopen(
#     "https://www.universalclass.com/articles/psychology/healthy-relationship-dynamics.htm")# 
# soup = BeautifulSoup(page.read(), 'html.parser')# html.parser html5lib


# table = soup.findAll(['body'])

# for row in table:
#     print(row.get_text())



# from urllib.request import urlopen
# from bs4 import BeautifulSoup
# import re

# pages = set()
# def getLinks(pageUrl):
#     global pages
#     html = urlopen('http://en.wikipedia.org{}'.format(pageUrl))
#     bs = BeautifulSoup(html, 'html.parser')
#     try:
#         print(bs.h1.get_text())
#         print(bs.find(id ='mw-content-text').find_all('p')[0])
#         print(bs.find(id='ca-edit').find('span').find('a').attrs['href'])
    
#     except AttributeError:
#         print('This page is missing something! Continuing.')

#     for link in bs.find_all('a', href=re.compile('^(/wiki/)')):
#         if 'href' in link.attrs:
#             if link.attrs['href'] not in pages:
#             #We have encountered a new page
#                 newPage = link.attrs['href']
#                 print('-'*20)
#                 print(newPage)
#                 pages.add(newPage)
#                 getLinks(newPage)
# getLinks('github.com')