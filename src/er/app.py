import os
import gc
import time
import pickle
import gradio as gr
from bm_runner import BM25Index
from evidence_retreiver_api import retreive_v1
from transformers import RobertaModel, RobertaTokenizer
from sem_search import embed_doc_from_bm25, top_n_closest, embed_text

_, _, evidence = retreive_v1('Love is a feeling')
print(_[0])
print(evidence[0])

# BM25 index preload and caching
pwd = os.getcwd()
model_path = pwd + '/models/'
print(model_path)
index = model_path + 'bm25_v1.pkl'
print(index)
os.listdir(model_path)

load_start = time.time()
the_index = pickle.load(open(index, 'rb'))



# Pretrained model preload and caching
model_version = 'distilroberta-base' # 'mrm8488/scibert_scivocab-finetuned-CORD19'  allenai/scibert_scivocab_uncased
do_lower_case = True
model = RobertaModel.from_pretrained(model_version)
tokenizer = RobertaTokenizer.from_pretrained(model_version, do_lower_case=do_lower_case)
load_time = time.time() - load_start
print("Index and pretrained model caching inference time: {:.2f} ms".format(load_time * 1000))


def retreive_v1(query):
    # Query search BM25 index
    search_start = time.time()
    doc_p = the_index.search(query, 10) 
    search_time = time.time() - search_start
    print("BM25 token index query search inference time: {:.2f} ms".format(search_time * 1000))

    paper_id = doc_p['_id'].to_list()
    url = doc_p['url'].to_list()
    evidence = doc_p['body'].to_list()

    # return paper_id, url, evidence
    return evidence[0]

def retreive_v2(query):
    # Query search BM25 index
    search_start = time.time()
    doc_p = the_index.search(query, 10) 
    search_time = time.time() - search_start
    print("BM25 token index query search inference time: {:.2f} ms".format(search_time * 1000))

    # Top3 semantic search for vector similarities using scibert finetuned-CORD19 embeddings and cosine similarity
    prior_time = time.time()
    docs, doc_embeds = embed_doc_from_bm25(doc_p, model, tokenizer)
    search_term_embeds = embed_text(query, model, tokenizer).mean(1)

    paper_id, url, evidence = top_n_closest(search_term_embeds, doc_embeds, docs, n=10)
    print("DistilRoBERTa semantic search inference time: {:.2f} ms".format((time.time() - prior_time) * 1000) )
    del query
    del doc_p
    del docs
    del doc_embeds
    del search_term_embeds
    gc.collect()


    return paper_id, url, evidence

ibot = gr.Interface(
  fn=retreive_v1, 
  inputs='text',
  outputs='text',
  examples=[["Love is a feeling"]]
)

ibot.launch()