import json
import requests
import pandas as pd
from ast import literal_eval
from sklearn.model_selection import train_test_split
from tqdm import tqdm
tqdm.pandas()


evidence_endpoint = "http://34.133.141.231:5020//v1/retreival" #http://35.225.9.207:5050/v1/retreival

def retreive_evidence(_id, query):

    body_data = { "_id" : _id, "query" : query }

    try:
        response = requests.post(evidence_endpoint, json=body_data) #,timeout=5)
        if response.status_code == 200:
            evidence = response.json()
            # response.close()
            # print(cred)
            paper_id,evidence = evidence['paper_id'], evidence['evidence']
            latency = response.elapsed.total_seconds()
            print(latency)
            return paper_id, evidence, latency
    # except UnboundLocalError as error:
    except UnicodeEncodeError as error:
        paper_id = 'NaN'
        evidence = 'NaN'
        latency = 0
        return paper_id, evidence, latency

# retreive_evidence( 'abc', 'Info that WHO does not recomend wearing masks.' )

df = pd.read_csv('gs://hackthon_covidliebusters/datasets/ifcn/ifcn_dataset.csv', nrows=100) # gs://hackthon_covidliebusters/datasets/tweets/covid_tweets.csv
print(list(df.columns.values))

df_ = pd.DataFrame(df.progress_apply(lambda row: retreive_evidence(row['_id'], row['claim']), axis=1).to_list(), columns =['paper_id', 'evidence', 'latency'])

df = pd.concat([df, df_], axis=1)

df.to_csv('gs://fake_news_corpus11_dev/cred_data/covidliebuster/latency_i.csv', sep='\t') 
latency_s = df['latency'].sum()
print(latency_s)





# wt = df['content'].progress_apply(extract)

# df['doc_wt'] = pd.Series(wt)

# df['content_'] = df['title']+ df['body_entity_sentiment'] + df['src_cred_label']  + df['content']
# df_ = pd.DataFrame(df.progress_apply(lambda row: str_wt(row['content_'], row['content']), axis=1).to_list(), columns =['class', 'confidence', 'doc_wt', 'latency'])

# df = pd.concat([df, df_], axis=1)

# df.to_csv('gs://fake_news_corpus11_dev/cred_data/latency_iii.csv', sep='\t')

# latency_s = df['latency'].sum()
# print(latency_s)