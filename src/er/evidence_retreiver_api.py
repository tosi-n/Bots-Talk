import os
import gc
import time
import pickle
from bm_runner import BM25Index
from flask import Flask, abort, request, jsonify
from transformers import RobertaModel, RobertaTokenizer
from sem_search import embed_doc_from_bm25, top_n_closest, embed_text



app = Flask(__name__)




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
model_version = 'distilroberta-base' 
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

    return paper_id, url, evidence
    # return evidence[0]


def retreive_v2(query):
    # Query search BM25 index
    search_start = time.time()
    doc_p = the_index.search(query, 10) 
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


    # outport = dict(zip(paper_id, evidence))
    return paper_id, url, evidence
@app.route("/")
def instructions():
    return 'EVIDENCE RETRIEVAL PIPELINE' \
           '\n' \
           '\nPOST a JSON file with input claims query text to retreive evidence'


@app.route('/health', methods=['GET'])
def health():
    if request.method == 'GET':
        print('health status OK')
        return 'ok'


@app.route('/v1/retreival', methods=['POST'])
def retrieval():
    # Step 1: Extract POST data from request body as JSON
    json_data = request.get_json(force=True)
    print(f'Inputs: {json_data}')
   
    # Check if required fields are in query
    if isinstance(json_data, dict):
        print('yes datatype confirmed')
        if 'query' not in json_data.keys() or '_id' not in json_data.keys():
            abort(400)

        query = json_data.get('query')

        # Step 2: Model prediction
        output = retreive_v2(query=query)
        paper_id, evidence_url, evidence = output[0], output[1], output[2]
        del output
        gc.collect()

        # # Step 3: Return the response as JSON
        # return jsonify({'_id': json_data['_id'],
        #                 'evidence': evidence,
        #                 'confidence': confidence})
        
        return jsonify({'_id': json_data['_id'],
                        # 'paper_id': paper_id,
                        'evidence_url': evidence_url,
                        'evidence': evidence})
    else:
        abort(400)

def create_app(port):
    app.run(debug=True, host='0.0.0.0', port=port, use_reloader=False)


if __name__ == "__main__":
    create_app(port=5020)# 8082 8081 5020
# jupyter-notebook --allow-root --ip 0.0.0.0 --port 5050