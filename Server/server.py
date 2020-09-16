import json
import os
import shutil
import random
import string
import pickle
import numpy as np
from flask import Flask, render_template, request, Response, abort
from flask_mail import Mail, Message
from flask_cors import CORS, cross_origin
from werkzeug.utils import secure_filename
from modules.EmbeddingsGenerator import EmbeddingsGenerator
from modules.SimilarityEstimator import SimilarityEstimator
from modules.SimilarityWrapper import SimilarityWrapper
from modules.Stemmer import get_common_words
from modules.TfIdf import TfIdf
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import check_password_hash, generate_password_hash
from modules.CommonNgrams import common_ngrams, common_ngram_txt


app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
db = SQLAlchemy(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(1000))
    password = db.Column(db.String(1000))
    isAdminUser = db.Column(db.Boolean)

db.create_all()

app.config.update(dict(
    DEBUG = True,
    MAIL_SERVER = 'smtp.gmail.com',
    MAIL_PORT = 587,
    MAIL_USE_TLS = True,
    MAIL_USE_SSL = False,
    MAIL_USERNAME = '',
    MAIL_PASSWORD = '',
))

mail = Mail(app)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

## Constants
UPLOAD_FOLDER = 'static/uploads'
OUTPUT_DIR = os.path.abspath('embeddings')

## Setup
embeddingsGenerator = EmbeddingsGenerator(delete_models=False)
similarityEstimator = SimilarityEstimator(output_dir=OUTPUT_DIR)
similarityWrapper = SimilarityWrapper(embeddingsGenerator, similarityEstimator, output_dir=OUTPUT_DIR)

def get_system_variables():
    if os.path.exists('static/uploads/data.csv'):
        print('Server: Reading Input File')
        input_csv = 'static/uploads/data.csv'
        try:
            with open('static/id_field_name.pickle', 'rb') as f:
                id_field_name = pickle.load(f)
                print('Server: Reading Saved ID Field Name = ', id_field_name)

            with open('static/sentence_field_name.pickle', 'rb') as f:
                sentence_field_name = pickle.load(f)
                print('Server: Reading Saved Sentence Field Name = ', sentence_field_name)

            with open('static/max_k.pickle', 'rb') as f:
                max_k = pickle.load(f)
                print('Server: Reading Saved Max K = ', max_k)

            with open('static/tfidf.pickle', 'rb') as f:
                tfIdf_model = pickle.load(f)
                print('Server: Reading tfIDF file', tfIdf_model)


        except Exception as e:
            print('Server: Error Resetting Field Names')
            id_field_name = ''
            sentence_field_name = ''
            max_k = 0
            tfIdf_model = None
    else:
        print('Server: No Input File')
        input_csv = ''
        id_field_name = ''
        sentence_field_name = ''
        max_k = 0
        tfIdf_model = None

    return {'input_csv': input_csv,
            'id_field_name': id_field_name,
            'sentence_field_name': sentence_field_name,
            'max_k': max_k,
            'tfIdf_model': tfIdf_model}

available_embeddings = similarityWrapper.get_available_embedding_algorithms()
available_similarity_metrics = similarityWrapper.get_available_distance_metrics()

def remove_substring_duplicates(sentences):
  filtered_sentences = []
  for sentence_A in sentences:
    found = False
    for sentence_B in sentences:
      if sentence_B == sentence_A:
        continue

      if sentence_A in sentence_B:
        found = True
        break

    if not found:
      filtered_sentences.append(sentence_A)

  return filtered_sentences


def get_sentence_variants(sentence):
  upper = sentence.upper()
  lower = sentence.lower()
  title = sentence.title()
  return [upper, lower, title]

def generate_list_variants(sentences):
  sentences_variants = []
  for sentence in sentences:
    if len(sentence) > 4:
        continue

    sentence_variants = get_sentence_variants(sentence)
    sentences_variants.extend(sentence_variants)

  sentences_variants.extend(sentences)
  return list(set(sentences_variants))

def upper_case_abbreviations(sentences):
    exclude = ['one', 'two', 'new', 'the', 'end', 'etc']
    sentences_ = []
    for sentence in sentences:
        sentences_.append(' '.join([word.upper() if len(word) <= 3 and word.lower() not in exclude else word for word in sentence.split()]))
    return sentences_

def remove_single_letter(sentences):
    return [sentence for sentence in sentences if len(sentence) > 1]

def exclude_from_auth(func):
    func._exclude_from_auth = True
    return func

def get_indexed():
    try:
        with open('static/indexed_embeddings.pickle', 'rb') as f:
            indexed_embeddings = pickle.load(f)

    except Exception as e:
        print('Server: ', e)
        indexed_embeddings = {}
    return indexed_embeddings

def add_index(embedding, metric):
    indexed_embeddings = get_indexed()
    indexed_embedding = indexed_embeddings.get(embedding, [])
    indexed_embedding.append(metric)
    indexed_embeddings[embedding] = indexed_embedding

    with open('static/indexed_embeddings.pickle', 'wb') as f:
        pickle.dump(indexed_embeddings, f)

@app.route('/email', methods=['POST'])
def email():
    msg = Message("Hello",
                  sender="from@example.com",
                  body = ' '.join(similarityWrapper.get_available_embedding_algorithms()),
                  recipients=[""])    

    mail.send(msg)
    return 'OK'

@exclude_from_auth
@app.route('/login', methods=['POST', 'GET'])
def login():
    name = request.form.get('name')
    password = request.form.get('password')

    user = User.query.filter_by(name=name).first()
    if not user or not check_password_hash(user.password, password):
        abort(403)

    return json.dumps({'isAdminUser': user.isAdminUser})


@app.route('/signup', methods=['POST'])
@exclude_from_auth
def signup():
    name = request.form.get('name')
    password = request.form.get('password')
    isAdminUser = True if (request.form.get('isAdminUser')) == 'true' else False

    user = User.query.filter_by(name=name).first()
    if user:
        abort(403)

    user = User(name = name, password = generate_password_hash(password, method='sha256'), isAdminUser=isAdminUser)

    db.session.add(user)
    db.session.commit()

    return 'OK'

@app.before_request
def before_callback():
    test_auth = True

    if request.endpoint in app.view_functions:
        view_func = app.view_functions[request.endpoint]
        test_auth = not hasattr(view_func, '_exclude_from_auth')

    if test_auth:
        if request.authorization:
            name = request.authorization.username
            password = request.authorization.password
            user = User.query.filter_by(name=name).first()

            if not user or not check_password_hash(user.password, password):
                abort(401)

        else:
            abort(401)


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
        user_id = request.form['id']
        K = int(request.form['k'])
        system_variables = get_system_variables()
        max_k = system_variables['max_k']
        K = K if K < max_k else max_k
        model_name = request.form['embedding']
        similarity_metric = request.form['similarity_metric']

        sentence, similar = similarityWrapper.get_most_similar_K_to(user_id, K, model_name, similarity_metric, system_variables['input_csv'])
        
        return render_template('search.html', obj = sentence, similar=similar, embeddings= available_embeddings, similarity_metrics = available_similarity_metrics)
    else:
        return render_template('search.html', embeddings = available_embeddings, similarity_metrics = available_similarity_metrics)

## Renders the web page to upload the csv file
@cross_origin()
@app.route('/upload')
def upload():
    return render_template('upload.html', embeddings = available_embeddings, similarity_metrics = available_similarity_metrics)


## GET Request: Returns the list of supported models
@cross_origin()
@app.route('/models', methods = ['GET'])
def get_models():
    return json.dumps(similarityWrapper.get_available_embedding_algorithms())


## GET Request: Returns the list of supported distance metrics
@cross_origin()
@app.route('/metrics', methods = ['GET'])
def get_metrics():
    return json.dumps(similarityWrapper.get_available_distance_metrics())


## GET Request: Returns the list of supported models
@cross_origin()
@app.route('/default_models', methods = ['GET'])
def get_default_models():
    default_embeddings = ['sentence_distilbert']
    return json.dumps(default_embeddings)


## GET Request: Returns the list of supported distance metrics
@cross_origin()
@app.route('/default_metrics', methods = ['GET'])
def get_default_metrics():
    indexed_metrics = ['cosine']
    return json.dumps(indexed_metrics)


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
    models = request.form.getlist('models')
    metrics = request.form.getlist('metrics')
    max_k = int(request.form['max_k'])
    chunk_size = int(request.form['chunk_size'])
    id_key = request.form['id_key']
    sentence_key = request.form['sentence_key']

    with open('static/id_field_name.pickle', 'wb') as f:
        print('Server: Writing ID Field Name..', id_key)
        pickle.dump(id_key, f)
        id_field_name = id_key

    with open('static/sentence_field_name.pickle', 'wb') as f:
        print('Server: Writing Sentence Field Name..', sentence_key)
        pickle.dump(sentence_key, f)
        sentence_field_name = sentence_key

    with open('static/max_k.pickle', 'wb') as f:
        print('Server: Writing max k = ',max_k)
        pickle.dump(max_k, f)

    f = request.files['file']
    file_path = '{}/{}'.format(UPLOAD_FOLDER, secure_filename('data.csv'))
    f.save(file_path)
    input_csv = file_path


    print('Server: Generating {} for {}'.format(models, metrics))
    
    try:
        print('Server: ', chunk_size, models, id_key, sentence_key, max_k, metrics, input_csv)
        _generate_models(chunk_size, models, id_key, sentence_key, max_k, metrics, input_csv)

        response =  Response(status=201)
    except KeyError as e:
        print('Server: ', e)
        response =  Response(status=422, response=json.dumps({'error': 'Key Error'}), mimetype='application/json')

    tfIdf_model = TfIdf()
    tfIdf_model.fit(input_csv, sentence_key)

    with open('static/tfidf.pickle', 'wb') as f:
        print('Server: Writing tfidf')
        pickle.dump(tfIdf_model, f)

    return response

## Post Request: Returns the objective of the given user is
## Params:
##      id: integer, the id of the user to return their objective
@cross_origin()
@app.route('/get_sentence', methods = ['POST'])
def get_sentence():
    try:
        system_variables = get_system_variables()
        input_csv = system_variables['input_csv']
        id_field_name = system_variables['id_field_name']
        if input_csv == '':
            return  Response(status=422, response=json.dumps({'error': 'Generate Embeddings First'}), mimetype='application/json')
        
        sentence =  similarityWrapper.get_sentence_at_id(request.json['id'], input_csv, id_field_name)
        return json.dumps({'sentence': sentence})

    except KeyError:
        return Response(status=422, response=json.dumps({'error': 'Key Error'}), mimetype='application/json')

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
    id = request.json['id']
    K = int(request.json['K'])
    system_variables = get_system_variables()
    max_k = system_variables['max_k']
    print('Server: comparing {} with {}'.format(K, max_k))
    K = K if K < max_k else max_k
    print('Server: ', method, metric, id, K)

    try:
        input_csv = system_variables['input_csv']
        id_field_name = system_variables['id_field_name']
        sentence_field_name = system_variables['sentence_field_name']

        user_objective = similarityWrapper.get_sentence_at_id(id, input_csv, id_field_name)
        __, nearest = similarityWrapper.get_most_similar_K_to(id, K, method, metric, input_csv, id_field_name, sentence_field_name)
        
        nearest = nearest.to_dict()
        other_objectives = list(nearest['sentence'].values())

        # Fix Spacing In Objectives
        for key, value in nearest['sentence'].items():
            nearest['sentence'][key] = ' '.join(value.split())
        user_objective = ' '.join(user_objective.split())

        for key, value in nearest['distance'].items():
            nearest['distance'][key] = int((1 - nearest['distance'][key] ) * 100)

        # other_objectives = [objective.lower() for objective in other_objectives]
        # nearest['sentence'] = other_objectives

        nearest['userObjective'] = user_objective
        
        # l1 += common_ngram_txt([user_objective], other_objectives, size=1)
        l1 = set()
        for other_objective in other_objectives:
            l_1 =  common_ngram_txt([user_objective], [other_objective], size=2) + common_ngram_txt([user_objective], [other_objective], size=3) + common_ngram_txt([user_objective], [other_objective], size=4)
            
            if len(l_1) < 2:
                l_1 += common_ngram_txt([user_objective], [other_objective], size = 1)[:3]
                
                if len(l_1) < 4:
                    l_1 += get_common_words([user_objective] + [other_objective])[:3]
            
            if len(l_1) >= 4:
                l_1.sort(key=len, reverse=True)
                l_1 = l_1[:4]

            for l__1 in l_1:
                if l__1.lower() not in l1 and l__1.upper() not in l1:
                    l1.add(l__1)
        
        # l1 = [l.lower() for l in l1]
        

        all_objectives = other_objectives + [user_objective]
        tfIdf_model = system_variables['tfIdf_model']
        # l2 = tfIdf_model.get_keys(all_objectives, n=40)
        l2 = set()
        for objective in all_objectives:
            l_2 = tfIdf_model.get_keys([objective], n=3)
            for l__2 in l_2:
                if l__2.lower() not in l2 or l__2.upper() not in l2:
                    l2.add(l__2)

        # l2 = remove_substring_duplicates(l2)
        # l2 = [l.lower() for l in l2]

        l1 = set([l.title() for l in l1])
        l2 = set([l.title() for l in l2])
        l1 = l1.difference(l2)
        # l1 = remove_substring_duplicates(l1)

        nearest['importantWords'] = np.unique(remove_single_letter(upper_case_abbreviations(list(l2)))).tolist()
        nearest['commonWords'] = np.unique(remove_single_letter(upper_case_abbreviations(list(l1)))).tolist()
 
        # nearest['importantWords'] = generate_list_variants(l2)
        # nearest['commonWords'] = generate_list_variants(l1)

        # generate_list_variants

        return json.dumps(nearest)
        
    except KeyError as e:
        print('Server: ', e)
        return Response(status=422, response=json.dumps({'error': 'Key Error'}), mimetype='application/json')

def _generate_models(chunk_size, models, id_key, sentence_key, max_k, metrics, input_csv, add_output_prefix=False):
    if os.path.exists('./embeddings'):
        shutil.rmtree('./embeddings')
    os.makedirs('./embeddings')

    output_prefix = ''.join(random.choices(string.ascii_uppercase, k=5)) if add_output_prefix else ''

    embedding_files = similarityWrapper.generate_emeddings_from_csv(input_csv, OUTPUT_DIR, output_prefix, chunk_size, models, id_key, sentence_key)
    for embedding_algorithm, embedding_file in embedding_files.items():
        similarityWrapper.create_embeddings_index(embedding_file, embedding_algorithm, max_k, metrics)


    if os.path.exists('static/indexed_embeddings.pickle'):
        os.remove('static/indexed_embeddings.pickle')

    for model in models:
        for metric in metrics:
            add_index(model, metric)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')