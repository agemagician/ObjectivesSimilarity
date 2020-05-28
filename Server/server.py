import json
import os
import random
import string
from flask import Flask, render_template, request, Response
from werkzeug.utils import secure_filename
from EmbeddingsGenerator import EmbeddingsGenerator
from SimilarityEstimator import SimilarityEstimator
from SimilarityWrapper import SimilarityWrapper

app = Flask(__name__)

## Constants
UPLOAD_FOLDER = 'static/uploads'
OUTPUT_DIR = os.path.abspath('embeddings')

## Setup
embeddingsGenerator = EmbeddingsGenerator(delete_models=False)
similarityEstimator = SimilarityEstimator(output_dir=OUTPUT_DIR)
similarity_server = SimilarityWrapper(embeddingsGenerator, similarityEstimator, output_dir=OUTPUT_DIR)
input_csv = os.path.abspath('static/uploads/minyset.csv')


@app.route('/', methods=['GET'])
def index():
    embeddings = similarity_server.get_available_embedding_algorithms()
    return render_template('index.html', embeddings=embeddings)


## Renders the web page to upload the csv file
@app.route('/upload')
def upload():
    embeddings = similarity_server.get_available_embedding_algorithms()
    similarity_metrics = similarity_server.get_available_distance_metrics()
    return render_template('upload.html', embeddings = embeddings, similarity_metrics = similarity_metrics)


## Uploads the csv file which will be used to generate the embeddings
@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
    print(request.form)
    if request.method == 'POST':
        selected_embeddings = request.form.getlist('embedding')
        selected_metrics = request.form.getlist('similarity_metric')

        f = request.files['file']
        file_path = '{}/{}'.format(UPLOAD_FOLDER, secure_filename(f.filename))
        f.save(file_path)
        input_csv = file_path
        return 'file uploaded successfully'


## GET Request: Returns the list of supported models
@app.route('/models', methods = ['GET'])
def get_models():
    return json.dumps(similarity_server.get_available_embedding_algorithms())


## GET Request: Returns the list of supported distance metrics
@app.route('/metrics', methods = ['GET'])
def get_metrics():
    return json.dumps(similarity_server.get_available_distance_metrics())

## Post Request: Generates Embeddings and Similarity Indices
## Params:
##      models: list of strings, the models to generate the embeddings for.
##      metrics: list of strings, the metrics to use in calculating the distance.
##      max_k: integer, the number of maximum similar users to index.
##      chunk_size: integer, chunk size to use in reading data from csv file
##      id_key: str, name of the column with the user ids
##      sentence_key: str, name of the column with user objectives
@app.route('/generate_models', methods = ['POST'])
def generate_models():
    models = request.json['models']
    metrics = request.json['metrics']
    max_k = request.json['max_k']
    chunk_size = request.json['chunk_size']
    id_key = request.json['id_key']
    sentence_key = request.json['sentence_key']

    print('Generating {} for {}'.format(models, metrics))
    
    output_prefix = ''.join(random.choices(string.ascii_uppercase, k=5))

    embedding_files = similarity_server.generate_emeddings_from_csv(input_csv, OUTPUT_DIR, output_prefix, chunk_size, models, id_key, sentence_key)
    for embedding_algorithm, embedding_file in embedding_files.items():
        similarity_server.create_embeddings_index(embedding_file, embedding_algorithm, max_k, metrics)

    return Response(status=201)

## Post Request: Returns the objective of the given user is
## Params:
##      id: integer, the id of the user to return their objective
@app.route('/get_sentence', methods = ['POST'])
def get_sentence():
    return similarity_server.get_sentence_at_id(request.json['id'], input_csv)

## Post Request: Returns the most similar users to the given id
## Params:
##      models: string, the embedding technique to use.
##      metric: string, the metric to use in calculating the distance.
##      K: integer, the number of similar users to return.
##      id: integer, the id of the user in query
## Returns:
##      list of user ids, objectives, and scores
@app.route('/get_nearest', methods = ['POST'])
def get_nearest():
    method = request.json['method']
    metric = request.json['metric']
    id = request.json['id']
    K = request.json['K']
    __, nearest = similarity_server.get_most_similar_K_to(id, K, method, metric, input_csv)
    return nearest.to_json()


if __name__ == '__main__':
   app.run(debug = True)