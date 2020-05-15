import pandas as pd
import random
import string

from tabulate import tabulate

class SimilarityServer:
  
  def __init__(self, embeddings_generator, similarity_estimator, output_dir = '/content/drive/My Drive'):
    self.embeddings_generator = embeddings_generator
    self.similarity_estimator = similarity_estimator
    self.output_dir = output_dir
    self.output_prefix = ''.join(random.choices(string.ascii_uppercase, k=5))
  
  def get_most_similar_K_to(self, id, K, model_name, similarity_metric):
    h5_file_name = '{}/{}_{}_index.h5'.format(self.output_dir, model_name, similarity_metric)
    hdf = pd.HDFStore(h5_file_name)

    key = 'key_{}'.format(id)
    similar_k = hdf.get(key).head(K).set_index('id')

    sentences = pd.read_csv(self.input_csv)
    sentences = sentences.set_index('id')

    similar_k = pd.concat([similar_k, sentences], axis=1, join='inner')

    print(sentences.loc[id][0])
    print(tabulate(similar_k.sort_values('distance')))

    return (sentences.loc[id], similar_k.sort_values('distance'))    


  def create_similarity_index(self, input_csv, model_names = None, similarity_metrics = None, chunk_size = 10):
    self.upload_csv(input_csv)
    embedding_files = self.generate_emeddings_from_csv(self.output_dir, self.output_prefix, chunk_size, model_names)
    for embedding_algorithm, embedding_file in embedding_files.items():
     self.create_embeddings_index(embedding_file, embedding_algorithm)

  def upload_csv(self, input_csv):
    self.input_csv = input_csv

  def get_available_embedding_algorithms(self):
    return self.embeddings_generator.get_supported_model_names()

  def get_available_distance_metrics(self):
    return self.similarity_estimator.get_available_distance_metrics()

  def generate_emeddings_from_csv(self, output_dir, output_prefix = '', chunk_size = 10, model_names = None):
    return self.embeddings_generator.save_embeddings_for_csv(self.input_csv, output_dir, output_prefix, chunk_size, model_names)

  def create_embeddings_index(self, hdf_file, embedding_algorithm, similarity_metrics = None):
    self.similarity_estimator.create_index_from_hdf(hdf_file, embedding_algorithm, similarity_metrics)