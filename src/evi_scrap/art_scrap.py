import feedparser as fp
import numpy as np
import json
import newspaper
from newspaper import Article, Config
from time import mktime
from datetime import datetime
import pandas as pd
import numpy as np
import csv
import time


user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'
config = Config()
config.memoize_articles = False
config.http_success_only = True
config.request_timeout = 1000000
# config.number_threads = 10
config.browser_user_agent = user_agent

# Set the limit for number of articles to download
LIMIT = 100000
articles_array = []

data = {}
data['newspapers'] = {}

# Loads the JSON files with news sites
with open('./src/al_data_config/etl_config/domain.json') as data_file:
    companies = json.load(data_file)


count = 1

# Iterate through each news company
for company, value in companies.items():
    # If a RSS link is provided in the JSON file, this will be the first choice.
    # Reason for this is that, RSS feeds often give more consistent and correct data. RSS (Rich Site Summary; originally RDF Site Summary; often called Really Simple Syndication) is a type of
    # web feed which allows users to access updates to online content in a standardized, computer-readable format
    # If you do not want to scrape from the RSS-feed, just leave the RSS attr empty in the JSON file.
    if 'rss' in value:
        d = fp.parse(value['rss'])
        print("Downloading articles from ", company)
        newsPaper = {
            "rss": value['rss'],
            "link": value['link'],
            "articles": []
        }
        for entry in d.entries:
            # Check if publish date is provided, if no the article is skipped.
            # This is done to keep consistency in the data and to keep the script from crashing.
            if hasattr(entry, 'published'):
                if count > LIMIT:
                    break
                article = {}
                article['link'] = entry.link
                date = entry.published_parsed
                article['published'] = datetime.fromtimestamp(mktime(date)).isoformat()
                try:
                    content = Article(entry.link)
                    content.download()
                    content.parse()
                except Exception as e:
                    # If the download for some reason fails (ex. 404) the script will continue downloading
                    # the next article.
                    print(e)
                    print("continuing...")
                    continue
                article['title'] = content.title
                article['text'] = content.text
                article['authors'] = content.authors
                article['top_image'] =  content.top_image
                article['movies'] = content.movies
                newsPaper['articles'].append(article)
                articles_array.append(article)
                print(count, "articles downloaded from", company, ", url: ", entry.link)
                count = count + 1
    else:
        # This is the fallback method if a RSS-feed domain is not provided.
        # It uses the python newspaper library to extract articles
        print("Building site for ", company)
        paper = newspaper.build(value['link'], config)
        newsPaper = {
            "link": value['link'],
            "articles": []
        }
        noneTypeCount = 0
        for content in paper.articles:
            if count > LIMIT:
                break
            try:
                content.download()
                content.parse()
            except Exception as e:
                print(e)
                print("continuing...")
                continue
            # Again, for consistency, if there is no found publish date the article will be skipped.
            # After 10 downloaded articles from the same newspaper without publish date, the company will be skipped.

            article = {}
            article['title'] = content.title
            article['authors'] = content.authors
            article['text'] = content.text
            article['top_image'] =  content.top_image
            article['movies'] = content.movies
            article['domain'] = content.url
            article['published'] = content.publish_date
            newsPaper['articles'].append(article)
            articles_array.append(article)
            print(count, "articles downloaded from", company, " using newspaper, url: ", content.url)
            count = count + 1
            #noneTypeCount = 0
    count = 1
    data['newspapers'][company] = newsPaper



#Finally it saves the articles as a CSV-file.
try:
    f = csv.writer(open('./src/al_data_config/etl_config/arti_scrap_v4.csv', 'w', encoding='utf-8'))
    f.writerow(['Title', 'Authors','Text'])
    #print(article)
    for artist_name in articles_array:
        title = artist_name['title']
        authors=artist_name['authors']
        text=artist_name['text']
        # image=artist_name['top_image']
        # video=artist_name['movies']
        # domain=artist_name['domain']
        # publish_date=artist_name['published']
        # Add each artist’s name and associated domain to a row
        f.writerow([title, authors, text])
except Exception as e: print(e)



df = pd.read_csv('./src/al_data_config/etl_config/arti_scrap_v4.csv')

print(len(df.index))

filename = "gs://fake_news_corpus11/article_scraped/" + "arti_scrap_" + time.strftime("%Y%m%d-%H%M") +".csv"
print(filename)

df.to_csv(filename, sep='\t', encoding='utf-8', index=False, header=True)