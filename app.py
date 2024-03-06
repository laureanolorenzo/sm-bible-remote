from flask import Flask, render_template, request, jsonify, url_for
from flask_cors import CORS
import json
from create_embeddings import *
from dotenv import load_dotenv
import requests
load_dotenv()
pc_API_key = os.getenv('PINECONE_API_KEY')
own_API_key = os.getenv('OWN_API_KEY')
doc_name = 'sb_test' #####CHANGE LATER
index_name='smart-bible'
pc_index = pinecone_index(pc_API_key,index_name) # Pinecone connection
pod_url = 'https://oz1tro5e97uz7t-8080.proxy.runpod.net'
app = Flask(__name__)
CORS(app)
app.config['JSON_AS_ASCII'] = False
@app.get('/')
def get_index():
    return render_template('index.html')
@app.post('/predict')
def post_predict():
    text = request.get_json().get('message') #Comes from the front end
    threshold = request.get_json().get('threshold')
    n_results_passed = request.get_json().get('n_results')
    if not threshold:
        threshold = -5
    if not n_results_passed:
        n_results_passed = 3   
    threshold = float(threshold)
    n_results_passed = int(n_results_passed)
    incoming_api_key = request.headers.get('Auth')
    if own_API_key != incoming_api_key:
        return jsonify({'status':401,'relevant_docs':[]})
    filtered_query = filter_stopwords(text)
    json_query = json.dumps({'message':filtered_query})
    answer = requests.post(
        pod_url + '/embed',
        data=json_query,
        headers={"Content-Type": "application/json"}
        )
    answer = answer.json()
    if answer['status'] != 200:
        return jsonify({'status':500,'relevant_documents':[]})
    else:
        status = 200
    vectors = answer['result']
    passages = semantic_search(vectors,pc_index)
    second_json_query = json.dumps({'passages':passages['passages'],'filtered_query':filtered_query})
    second_answer = requests.post(
    pod_url + '/predict',
    data=second_json_query,
    headers={"Content-Type": "application/json"}
    )
    second_answer = second_answer.json()
    sorted_passages = second_answer['cross_scores']
    second_result = get_second_results(sorted_passages,threshold=threshold,mapped_results=passages)
    relevant_docs = get_context(second_result,n_results_passed)
    if not len(relevant_docs):
        status = 201
        relevant_docs = []
    response = {
        'status':status,
        'relevant_documents': relevant_docs, #etc
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run()