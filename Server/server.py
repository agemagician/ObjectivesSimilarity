import json
import os
import random
import string
import pickle
from flask import Flask, render_template, request, Response
from flask_cors import CORS, cross_origin
from werkzeug.utils import secure_filename
from modules.EmbeddingsGenerator import EmbeddingsGenerator
from modules.SimilarityEstimator import SimilarityEstimator
from modules.SimilarityWrapper import SimilarityWrapper

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

## Constants
UPLOAD_FOLDER = 'static/uploads'
OUTPUT_DIR = os.path.abspath('embeddings')

## Setup
embeddingsGenerator = EmbeddingsGenerator(delete_models=False)
similarityEstimator = SimilarityEstimator(output_dir=OUTPUT_DIR)
similarity_server = SimilarityWrapper(embeddingsGenerator, similarityEstimator, output_dir=OUTPUT_DIR)
input_csv = os.path.abspath('static/uploads/tinyset.csv')


available_embeddings = similarity_server.get_available_embedding_algorithms()
available_similarity_metrics = similarity_server.get_available_distance_metrics()


def get_indexed():
    try:
        with open('indexed_embeddings.pickle', 'rb') as f:
            indexed_embeddings = pickle.load(f)

    except Exception as e:
        print(e)
        indexed_embeddings = {}

    return indexed_embeddings

def add_index(embedding, metric):
    indexed_embeddings = get_indexed()
    indexed_embedding = indexed_embeddings.get(embedding, [])
    indexed_embedding.append(metric)
    indexed_embeddings[embedding] = indexed_embedding

    with open('indexed_embeddings.pickle', 'wb') as f:
        print('Writing..', indexed_embeddings)
        pickle.dump(indexed_embeddings, f)



@app.after_request
def after_request(response):
   response.headers.add('Access-Control-Allow-Origin', '*')
   response.headers.add('Access-Control-Allow-Headers', 'Content-  Type,Authorization')
   response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE')
   return response
   
## Renders the Home Page
@app.route('/')
def index():
    return render_template('index.html')


## Renders the page to query for similar users
@cross_origin()
@app.route('/search', methods=['GET', 'POST'])
def search():
    if request.method == 'POST':
        user_id = int(request.form['id'])
        K = int(request.form['k'])
        model_name = request.form['embedding']
        similarity_metric = request.form['similarity_metric']

        print(similarity_metric, model_name)

        sentence, similar = similarity_server.get_most_similar_K_to(user_id, K, model_name, similarity_metric, input_csv)
        
        print(similar.keys())
        
        return render_template('search.html', obj = sentence, similar=similar, embeddings= available_embeddings, similarity_metrics = available_similarity_metrics)
    else:
        return render_template('search.html', embeddings = available_embeddings, similarity_metrics = available_similarity_metrics)

## Renders the web page to upload the csv file
@cross_origin()
@app.route('/upload')
def upload():
    return render_template('upload.html', embeddings = available_embeddings, similarity_metrics = available_similarity_metrics)


## Uploads the csv file which will be used to generate the embeddings
@cross_origin()
@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
    print(request.form)
    if request.method == 'POST':
        selected_embeddings = request.form.getlist('embedding')
        selected_metrics = request.form.getlist('similarity_metric')

        max_k = int(request.form['max_k'])
        chunk_size = int(request.form['chunk_size'])

        id_key = request.form['id_key']
        sentence_key = request.form['sentence_key']

        f = request.files['file']
        file_path = '{}/{}'.format(UPLOAD_FOLDER, secure_filename(f.filename))
        f.save(file_path)
        input_csv = file_path

        _generate_models(chunk_size, selected_embeddings, id_key, sentence_key, max_k, selected_metrics, input_csv)
        # print(chunk_size, selected_embeddings, id_key, sentence_key, max_k, selected_metrics, input_csv)
        return 'Embeddings Generated Successfully'


## GET Request: Returns the list of supported models
@cross_origin()
@app.route('/models', methods = ['GET'])
def get_models():
    return json.dumps(similarity_server.get_available_embedding_algorithms())


## GET Request: Returns the list of supported distance metrics
@cross_origin()
@app.route('/metrics', methods = ['GET'])
def get_metrics():
    return json.dumps(similarity_server.get_available_distance_metrics())


## GET Request: Returns the list of supported models
@cross_origin()
@app.route('/indexed_models', methods = ['GET'])
def get_indexed_models():
    indexed_embeddings = list(get_indexed().keys())
    return json.dumps(indexed_embeddings)


## GET Request: Returns the list of supported distance metrics
@cross_origin()
@app.route('/indexed_metrics', methods = ['GET'])
def get_indexed_metrics():
    indexed_metrics = list(get_indexed().values())
    flatten = lambda l: [item for sublist in l for item in sublist]
    indexed_metrics = list(set(flatten(indexed_metrics)))
    return json.dumps(indexed_metrics)


## Post Request: Generates Embeddings and Similarity Indices
## Params:
##      models: list of strings, the models to generate the embeddings for.
##      metrics: list of strings, the metrics to use in calculating the distance.
##      max_k: integer, the number of maximum similar users to index.
##      chunk_size: integer, chunk size to use in reading data from csv file
##      id_key: str, name of the column with the user ids
##      sentence_key: str, name of the column with user objectives
@cross_origin()
@app.route('/generate_models', methods = ['POST'])
def generate_models():
    models = request.json['models']
    metrics = request.json['metrics']
    max_k = int(request.json['max_k'])
    chunk_size = int(request.json['chunk_size'])
    id_key = request.json['id_key']
    sentence_key = request.json['sentence_key']

    print('Generating {} for {}'.format(models, metrics))
    
    print(chunk_size, models, id_key, sentence_key, max_k, metrics, input_csv)
    _generate_models(chunk_size, models, id_key, sentence_key, max_k, metrics, input_csv)

    return Response(status=201)

## Post Request: Returns the objective of the given user is
## Params:
##      id: integer, the id of the user to return their objective
@cross_origin()
@app.route('/get_sentence', methods = ['POST'])
def get_sentence():
    sentence =  similarity_server.get_sentence_at_id(int(request.json['id']), input_csv)
    return json.dumps({'sentence': sentence})

## Post Request: Returns the most similar users to the given id
## Params:
##      models: string, the embedding technique to use.
##      metric: string, the metric to use in calculating the distance.
##      K: integer, the number of similar users to return.
##      id: integer, the id of the user in query
## Returns:
##      list of user ids, objectives, and scores
@cross_origin()
@app.route('/get_nearest', methods = ['POST'])
def get_nearest():
    method = request.json['method']
    metric = request.json['metric']
    id = int(request.json['id'])
    K = int(request.json['K'])
    print(method, metric, id, K)

    __, nearest = similarity_server.get_most_similar_K_to(id, K, method, metric, input_csv)
    return nearest.to_json()


def _generate_models(chunk_size, models, id_key, sentence_key, max_k, metrics, input_csv, add_output_prefix=False):
    output_prefix = ''.join(random.choices(string.ascii_uppercase, k=5)) if add_output_prefix else ''

    embedding_files = similarity_server.generate_emeddings_from_csv(input_csv, OUTPUT_DIR, output_prefix, chunk_size, models, id_key, sentence_key)
    for embedding_algorithm, embedding_file in embedding_files.items():
        similarity_server.create_embeddings_index(embedding_file, embedding_algorithm, max_k, metrics)

    for model in models:
        for metric in metrics:
            add_index(model, metric)


if __name__ == '__main__':
   app.run(debug = True)