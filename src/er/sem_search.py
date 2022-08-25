import gc
import math
import nltk
import torch
import string
import collections
import numpy as np
import pandas as pd
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
tqdm.pandas()


"""
model_version = 'allenai/scibert_scivocab_uncased' # mrm8488/scibert_scivocab-finetuned-CORD19
do_lower_case = True
model = BertModel.from_pretrained(model_version)
tokenizer = BertTokenizer.from_pretrained(model_version, do_lower_case=do_lower_case)
"""

def embed_text(text, model, tokenizer):
    try:
        input_ids = torch.tensor(tokenizer.encode(text)).unsqueeze(0)  # Batch size 1
        # outputs = 
        last_hidden_states = model(input_ids)[0]  # The last hidden-state is the first element of the output tuple
        del input_ids
        gc.collect()
    except:
        # outputs = np.NaN
        last_hidden_states = np.NaN
    return last_hidden_states 

def get_similarity(em, em2):
    return cosine_similarity(em.detach().numpy(), em2.detach().numpy())

def top_n_closest(search_term_embedding, doc_embeddings, doc_paragraphs, n=10): #_id_n_doc
    proximity_dict = {}
    i = 0 
    for doc_embedding in doc_embeddings:
        # print(doc_embedding)
        proximity_dict[doc_paragraphs[i]] = {"score": get_similarity(doc_embedding.unsqueeze(0),search_term_embedding), "doc_embedding":doc_embedding.unsqueeze(0)}
        i+=1
    order_dict = collections.OrderedDict({k: v for k, v in sorted(proximity_dict.items(), key=lambda item: item[1]["score"])})
    del proximity_dict
    proper_list = list(order_dict.keys())[-n:]
    # return proper_list, order_dict
    del order_dict
    paper_id = []
    url = []
    evidence = []
    for doc in proper_list:
        # print(doc)
        split_ = str(doc).split(" <||> ")
        paper_id.append(split_[0])
        url.append(split_[1])
        evidence.append(split_[2])
    del proper_list
    gc.collect()
    return paper_id, url, evidence

def is_ascii(s):
    try:
        s.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True

# def embed_doc_from_bm25(docs, model, tokenizer):
#     # docs = docs["body_text"].tolist()
#     # docs["body"] = docs[docs.body_text.map(lambda x: x.isascii())]
#     docs["_id_n_doc"] = docs["_id"] + " <||> " + docs["body"]
#     # docs["_id_n_doc"] = docs["_id"] + " <||> " + docs["url"] + " <||> " + docs["body"]
#     # print(len(docs.index))
#     docs = docs["_id_n_doc"].tolist()
#     # print(docs[0])
    
#     doc_embed = []
#     for doc in docs:
#         if len(tokenizer.encode(doc))<512:
#             # print(len(tokenizer.encode(doc)))
#             doc_embed.append(embed_text(doc, model, tokenizer).mean(1).squeeze(0))
#         else: 
#             # print(len(tokenizer.encode(doc)))
#             # TO-DO truncate to max allowable length
#             doc2 = " ".join(doc.split()[:200])
#             # print(len(doc2.split()))
#             doc_embed.append(embed_text(doc2, model, tokenizer).mean(1).squeeze(0))
#     doc_embed = [x for x in doc_embed if math.isnan(x) == False]
#     return docs, doc_embed #_id_n_doc/docs

def sent_token(doc):
    return nltk.sent_tokenize(doc)

def embed_doc_from_bm25(docs, model, tokenizer):
    # docs = docs["body_text"].tolist()
    
    docs["lang_f"] = docs.body.apply(is_ascii)#[doc_paragragh.body.map(lambda x: x.isascii())]
    docs = docs[docs['lang_f'] == True]
    docs['tokenized_sent'] = docs['body'].progress_apply(sent_token)
    lst_col = 'tokenized_sent'
    print(len(docs.index))
    docs = pd.DataFrame({col:np.repeat(docs[col].values, docs[lst_col].str.len()) for col in docs.columns.drop(lst_col)}).assign(**{lst_col:np.concatenate(docs[lst_col].values)})[docs.columns]
    print(len(docs.index))
    docs = docs.sample(frac=1)
    docs = docs.head(20)
    print(len(docs.index))
    # docs["_id_n_doc"] = docs["_id"] + " <||> " + docs["body"]
    docs["_id_n_doc"] = docs["_id"] + " <||> " + docs["url"] + " <||> " + docs["tokenized_sent"]
    
    docs = docs["_id_n_doc"].tolist()
    # print(docs[0])

    doc_embed = []
    for doc in docs:
        try:
            # if len(tokenizer.encode(doc))<512:
                # print(len(tokenizer.encode(doc)))
            # doc = 
            d_e = embed_text(" ".join(doc.split()[:200]), model, tokenizer)
            # print(d_e)
            # if d_e == np.NaN: raise AttributeError()
            embed = d_e.mean(1).squeeze(0)
            del d_e
            doc_embed.append(embed)
            del embed
            gc.collect()
        except Exception as e:
            embed = np.NaN
            doc_embed.append(embed)
            # else: 
                # print(len(tokenizer.encode(doc)))
                # TO-DO truncate to max allowable length
                
                # print(len(doc2.split()))
                # print(doc)
    doc_embed = pd.DataFrame([doc_embed]).T
    doc_embed.dropna(inplace=True)
    doc_embed = doc_embed[0].to_list()            
    # doc_embed = [x for x in doc_embed if math.isnan(x) == False]
    # doc_embed = [x for x in doc_embed if x != np.NaN]
    return docs, doc_embed #_id_n_doc/docs






# from ast import literal_eval

# def paragraphize_body(body_text):
#     paragraphs = [di['text'] for di in body_text if len(di['text'].split()) > 1]
#     return paragraphs

# df = pd.read_csv('gs://fake_news_corpus11_dev/cred_data/covidliebuster/cov_evi_combined_ii.csv', sep='\t', nrows=5)
# text = literal_eval(df[:1]['body_text'].to_list()[0])
# text = "\t \\\ ".join(text)
# print(text)
# for i in text:
#     print(i)