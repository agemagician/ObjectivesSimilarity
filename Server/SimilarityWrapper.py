from SimilarityEstimator import SimilarityEstimator
from EmbeddingsGenerator import EmbeddingsGenerator
import pandas as pd
import random
import string

from tabulate import tabulate

class SimilarityWrapper:
  """
  Prelimenary backend server to drive the whole process.
  """

  def __init__(self,
    embeddings_generator: EmbeddingsGenerator,
    similarity_estimator: SimilarityEstimator,
    output_dir = '/content/drive/My Drive'):
    """
    Server takes an instance of the embeddings generator class, and and instance of the similarity estimator class.

    Arguments:
        embeddings_generator {EmbeddingsGenerator} -- instance of the embeddings generator class
        similarity_estimator {SimilarityEstimator} -- instance of the similarity estimator class

    Keyword Arguments:
        output_dir {str} -- location of the folder to write the files to (default: {'/content/drive/My Drive'})
    """
    self.embeddings_generator = embeddings_generator
    self.similarity_estimator = similarity_estimator
    self.output_dir = output_dir
    self.output_prefix = ''.join(random.choices(string.ascii_uppercase, k=5))
  
  def get_sentence_at_id(self, id, input_csv):
    sentences = pd.read_csv(input_csv)
    sentences = sentences.set_index('id')

    return sentences.loc[id][0]


  def get_most_similar_K_to(self, id, K, model_name, similarity_metric, input_csv):
    """
    Returns the most similar ids to the given id. Prints the result.

    Arguments:
        id {str} -- id of the user in question.
        K {int} -- number of nearest users to retrieve
        model_name {str} -- name of the model to use for the embeddings
        similarity_metric {str} -- name of the similarity metric to calculate the similarity
        input_csv {str} -- path to the csv file containing the sentences  

    Returns:
        tuple -- returns the objective of the user in question, and a dataframe of the neares k users
    """
    h5_file_name = '{}/{}_{}_index.h5'.format(self.output_dir, model_name, similarity_metric)
    hdf = pd.HDFStore(h5_file_name)

    key = 'key_{}'.format(id)
    print(K)
    similar_k = hdf.get(key).head(K).set_index('id')

    sentences = pd.read_csv(input_csv)
    sentences = sentences.set_index('id')
    
    print(sentences)
    print(similar_k)

    similar_k = pd.concat([similar_k, sentences], axis=1, join='inner')

    print(sentences.loc[id][0])
    print(tabulate(similar_k.sort_values('distance')))

    return (sentences.loc[id], similar_k.sort_values('distance'))    


  def create_similarity_index(self, input_csv, model_names = None, similarity_metrics = None, chunk_size = 10, max_k = 10):
    """
    Drives the end-to-end process. Takes the input sentences in the input_csv, and indexes the similarity scores in the hd5 files.

    Arguments:
        input_csv {str} -- path to the csv file containing the sentences  

    Keyword Arguments:
        model_names {list} -- list of the model names to use in getting the embeddings. Uses all if None. (default: {None})
        similarity_metrics {list} -- list of the similarity metrics to use in calculating the distance. Uses all if None. (default: {None})
        chunk_size {int} -- chunk size (default: {10})
    """
    embedding_files = self.generate_emeddings_from_csv(self.output_dir, self.output_prefix, chunk_size, model_names)
    for embedding_algorithm, embedding_file in embedding_files.items():
     self.create_embeddings_index(embedding_file, embedding_algorithm, max_k= max_k)

  def get_available_embedding_algorithms(self):
    """
    Returns the list of embedding methods supported by the system.
    """
    return self.embeddings_generator.get_supported_model_names()

  def get_available_distance_metrics(self):
    """
    Returns the list of similarity techniques supported by the system.
    """
    return self.similarity_estimator.get_available_distance_metrics()

  def generate_emeddings_from_csv(self, input_csv, output_dir, output_prefix = '', chunk_size = 10, model_names = None, id_key = 'id', sentence_key = 'sentence'):
    """
    Takes the input sentences in the input_csv, and generates embeddings
    Arguments:
        input_csv {str} -- path to the csv file containing the sentences  
        output_dir {str} -- path to the directory to which the embeddings should be written
        output_prefix {str} -- prefix to the embedding file names

    Keyword Arguments:
        chunk_size {int} -- chunk size (default: {10})
        model_names {list} -- list of the model names to use in getting the embeddings. Uses all if None. (default: {None})
    """
    return self.embeddings_generator.save_embeddings_for_csv(input_csv, output_dir, output_prefix, chunk_size, model_names, id_key, sentence_key)

  def create_embeddings_index(self, hdf_file, embedding_algorithm, max_k, similarity_metrics = None):
    """
    Reads embeddings from an embeddings file, and creates similarity indexes between sentences
    Arguments:
      hdf_file {str} -- path to an hdf5 with the embeddings.
      embedding_algorithm {str} -- name of the embeddings algorithm of choice
      max_k {int} -- maximum number of similar sentences to index
      
    Keyword Arguments:
      similarity_metrics {list} -- names of the similarity metrics to use. Defaults to all metrics.
    """
    self.similarity_estimator.create_index_from_hdf(hdf_file, embedding_algorithm, max_k, similarity_metrics)