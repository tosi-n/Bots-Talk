# data loading and storage
from operator import index
import pandas as pd
import numpy as np
import os
from pathlib import Path
import json
from  copy import deepcopy
import pickle
import gc
import re

# preprocessing
import string
import nltk
nltk.download('stopwords')
nltk.download('punkt')

# model
from rank_bm25 import BM25Okapi

# other
from tqdm import tqdm
tqdm.pandas()

from argparse import ArgumentParser

pwd = os.getcwd()
# resource_path = pwd + "/resource"
model_path = pwd + '/models/'

# if not os.path.exists(resource_path):
#     os.mkdir(resource_path)

# if not os.path.exists(model_path):
    # os.mkdir(model_path)




print("Current working directory is:", os.getcwd())

model_path = pwd + '/models/'
if not os.path.exists(model_path):
    # cmd = 'gsutil -m cp gs://fake_news_corpus11_dev/cred_data/covidliebuster/models/bm25_v1.pkl'+ ' ' + model_path
    os.mkdir(model_path)
    print('Downloading Index...')
    # os.system(cmd)
    os.listdir(model_path)
else:
    print('Index already present...')
    os.listdir(model_path)


# varify the path using getcwd()
# cwd = os.getcwd()
# # print the current directory
# print("Current working directory is:", cwd)

os.chdir(pwd) 

"""
Sets up a BM25 index over the abstracts and results of the data.
"""

PUNCTUATION_REMOVER = str.maketrans('', '', string.punctuation)
STOPWORDS = set(nltk.corpus.stopwords.words('english'))



def paragraphize_body(body_text):
    paragraphs = [di['text'] for di in body_text if len(di['text'].split()) > 1]
    return paragraphs

def remove_newline(text):

    return text.strip('\n').replace('\n',' ').replace('»', ' ').translate(PUNCTUATION_REMOVER).lower() #re.sub('\r\n', '', text)


class BM25Index:
    def __init__(self, df):
        df['body'] = df['body'].progress_apply(remove_newline)
        self.data = df
        self.clean_data = self.data.body.progress_apply(clean_text).tolist()
        self.index = BM25Okapi(self.clean_data)
        # self.clean_data = df['clean_text'].tolist()
        # self.index = BM25Okapi(self.clean_data)

    def search(self, query, k=10):
        processed = clean_text(query)
        doc_scores = self.index.get_scores(processed)
        del processed
        ind = np.argsort(doc_scores)[::-1][:k]
        # df['body'] = df['body'].progress_apply(remove_newline)
        # self.data = df
        results = self.data.iloc[ind].copy()
        results['score'] = doc_scores[ind]
        del doc_scores
        del ind
        gc.collect()
        return results

def clean_text(text):
    print(text)
    uncased = text.translate(PUNCTUATION_REMOVER).lower()
    tokens = [token for token in nltk.word_tokenize(uncased) 
                if len(token) > 1
                and not token in STOPWORDS
                and not (token.isnumeric() and len(token) != 4)
                and (not token.isnumeric() or token.isalpha())]
    
    return tokens

if __name__ == '__main__':
    psr = ArgumentParser()
    psr.add_argument("--rebuild-index", action='store_true', default=False)
    psr.add_argument("--index-path", type=str, default="/Users/lade/Documents/Dev/Side_projects/Bots Talk/models/bm25_v1.pkl")#./models/bm25_v1.pkl
    psr.add_argument("--result-path-base", type=str, default="/Users/lade/Documents/Dev/Side_projects/Bots Talk/data/query") #./data/query
    psr.add_argument("--query", type=str)
    psr.add_argument("--nresults", type=int, default=5)
    psr.add_argument("--paragraphs", action='store_true', default=True)
    args = psr.parse_args()

    if args.rebuild_index or not os.path.isfile(args.index_path):
        # files = load_files(args.data_dir)
        # print("Loaded {} files".format(len(files)))
        df = pd.read_csv('/Users/lade/Documents/Dev/Side_projects/Bots Talk/data/evidence_ep1.csv', sep='\t')# , nrows=400000 #./data/evidence_ep1.csv
        df = df[['_id', 'url', 'body']]
        df['body'] = df['body'].progress_apply(remove_newline)
        search_idx = BM25Index(df)
        del df
        gc.collect()
        print("Caching index...")
        pickle.dump(search_idx, open(args.index_path, "wb"))
        print("Done caching...")
    else:
        print("Loading cached index...")
        search_idx = pickle.load(open(args.index_path, "rb"))
    args.query = 'Love is a feeling'
    results = search_idx.search(args.query)
    # print(results[['paper_id', 'title', 'body_text', 'text', 'score']])
    print(results[['_id', 'url', 'body', 'score']])
    results.to_csv("_".join([args.result_path_base, args.query.replace(" ","_"), "top{}".format(args.nresults)]) + ".csv", index=False)
