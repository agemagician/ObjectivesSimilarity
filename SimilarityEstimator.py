import numpy as np
import tensorflow as tf
import pandas as pd

from scipy.spatial import distance
from sklearn.preprocessing import minmax_scale
from utils import assert_row_dim, order_keys, get_non_dupicate_keys, update_heaps
from keyValueHeap import keyValueHeap


class SimilarityEstimator:
    """
    Handles calculating the similarity between embeddings. Stores the calculated similarities in hd5 file or returns them.
    """

    def __init__(self):

        ## TODO: Remove this directory when migrating from google colab
        self.output_dir = '/content/drive/My Drive'

        self.distances = {
            'cosine': distance.cosine,
            'minkowski': distance.minkowski,
            'correlation': distance.correlation,
            'jensenshannon': distance.jensenshannon,
            'cityblock': distance.cityblock,
            'sqeuclidean': distance.sqeuclidean,
            'euclidean': distance.euclidean,
            'chebyshev': distance.chebyshev,
            'canberra': distance.canberra,
            'braycurtis': distance.braycurtis
        }

    def get_available_distance_metrics(self):
        """
        Returns the available distance metrics

        Returns: list
        """
        return list(self.distances.keys())

    def _get_distance_functions(self, distance_metrics):
        """
        converts a list of distance metric names to distance functions.

        Arguments:
            distance_metrics {list} -- list of distance metric names. Must be from the available distance metrics.

        Returns:
            [dict] -- Dictionary of the form {distance_metric: distance_function}
        """
        if distance_metrics == None:
            return self.distances
        else:
            return {distance_metric: self.distances[distance_metric] for distance_metric in distance_metrics}

    def create_index_from_hdf(self, hdf_file, embedding_algoirthm, distance_metrics = None):
        """
        Reads the embeddings stored in an hd5 file. Calculates the similarity and indexes them in a separate hd5 file for each distance metric. 

        Arguments:
            hdf_file {str} -- path to the hd5 fie with the generated embeddings from EmbeddingsGenerator
            embedding_algoirthm {str} -- name of the model used to embed the sentences from EmbeddingsGenerator

        Keyword Arguments:
            distance_metrics {list} -- list of the distance metrics to use when calculating the similarity.  (default: {None})
                                    If none, will be initialized to the list of available distance metrics.
        """
        distance_functions = self._get_distance_functions(distance_metrics)

        print('Opening ', hdf_file)
        hdf = pd.HDFStore(hdf_file, mode='r')

        try:
            self.__loop_over_chunks(hdf, embedding_algoirthm, distance_functions)
        finally:
            hdf.close()
            print('Closing ', hdf_file)

    def __loop_over_chunks(self, hdf, embedding_algoirthm, distance_functions):
        """
        Helper used by create_index_from_hdf. Iterates over chunks and calculates similarity.

        Arguments:
            hdf {str} -- path to the hd5 file which contains the embeddings.
            embedding_algoirthm {str} -- name of the model used to generate the embeddings.
            distance_functions {list} -- list of the distance functions to use in calculating the similarity.
        """
        for chunk1_key in hdf.keys():
            chunk1 = hdf.get(chunk1_key)
            
            distances_results = dict()
            print('\nStarting {} Chunk {}'.format(embedding_algoirthm, chunk1_key))
            for chunk2_key in hdf.keys():
                chunk2 = hdf.get(
                    chunk2_key) if chunk2_key != chunk1_key else chunk1
                self.__calculate_chunk_chunk_similarity(
                    chunk1, chunk2, distance_functions, distances_results)

            self.__write_chunk_distances(distances_results, embedding_algoirthm, chunk1.index)
            print('Done with {} Chunk {}\n'.format(embedding_algoirthm, chunk1_key))        

    def __calculate_chunk_chunk_similarity(self, chunk1, chunk2, distance_functions, distances_results):
        """
        Handles the calculation of similarity between two chunks.

        Arguments:
            chunk1 {pd.DataFrame} -- pandas dataframe representing the first chunk
            chunk2 {pd.DataFrame} --  pandas dataframe representing the second chunk
            distance_functions {list} -- list of the distance functions to use in calculating the similarity 
            distances_results {dict} -- dictionary of the form {distance_metric: { id: keyValueHeap }}. Used to keep track 
                                        of the data which will be eventually written to hd5 files.
        """
        chunk1_row_keys, chunk2_row_keys = chunk1.index, chunk2.index
        keys_to_compare = get_non_dupicate_keys(
            chunk1_row_keys, chunk2_row_keys)

        for row1_key, row2_key in keys_to_compare:
            for distance_metric, distance_function in distance_functions.items():

                row1 = chunk1.loc[row1_key]
                assert_row_dim(row1.ndim, row1_key)

                row2 = chunk2.loc[row2_key]
                assert_row_dim(row2.ndim, row2_key)

                similarity = (-1) * distance_function(row1, row2)
                update_heaps(distances_results, distance_metric,
                             row1_key, row2_key, similarity)

    def __write_chunk_distances(self, distances_result, embedding_algoirthm, ready_keys):
        """
        Helper used by __calculate_chunk_chunk_similarity. Writes the accumulated results from distances_result into an hd5 file.

        Arguments:
            distances_result {dict} -- 
            embedding_algoirthm {str} -- name of the model used to generate the embeddings
            ready_keys {list} -- list of the keys that already have their nearest keys identified.
        """
        for distance_metric, distance_results in distances_result.items():
            store_name = '{}/{}_{}_index.h5'.format(self.output_dir, embedding_algoirthm, distance_metric)

            print('Writing To {}'.format(store_name), end = '... ')
            store = pd.HDFStore(store_name)

            for key, results in distance_results.items():
                if key not in ready_keys: continue
                ordered_results_df = pd.DataFrame(keyValueHeap.reverse(results), columns = ['id', 'distance'])

                store.append('key_{}'.format(key), ordered_results_df)

            store.close()
            print('Done Writing {} data'.format(distance_metric))

    def cosine(self, data):
        """
        Calculates cosine distance

        Arguments:
            data {list} -- list of sentences.

        Returns:
            np.array -- similarity matrix of input sentences.
        """
        return self.__common_routine(data, self.distances['cosine'])

    def minkowski(self, data):
        """
        Calculates minkowski distance

        Arguments:
            data {list} -- list of sentences.

        Returns:
            np.array -- similarity matrix of input sentences.
        """        
        return self.__common_routine(data, self.distances['minkowski'])

    def correlation(self, data):
        """
        Calculates correlation of input data

        Arguments:
            data {list} -- list of sentences.

        Returns:
            np.array -- similarity matrix of input sentences.
        """        
        return self.__common_routine(data, self.distances['correlation'])

    # Does not work if normalize is False
    # TODO: Normalize data before reaching this step
    def jensen_shannon(self, data, normalize=True):
        """
        Calculates jensen shannon distance

        Arguments:
            data {list} -- list of sentences.

        Returns:
            np.array -- similarity matrix of input sentences.
        """        
        return self.__common_routine(data, self.distances['jensenshannon'])

    def cityblock_distance(self, data):
        """
        Calculates cityblock distance

        Arguments:
            data {list} -- list of sentences.

        Returns:
            np.array -- similarity matrix of input sentences.
        """        
        return self.__common_routine(data, self.distances['cityblock'])

    def squared_euclidian_distance(self, data):
        """
        Calculates squared euclidean distance

        Arguments:
            data {list} -- list of sentences.

        Returns:
            np.array -- similarity matrix of input sentences.
        """        
        return self.__common_routine(data, self.distances['sqeuclidean'])

    def euclidean_distance(self, data):
        """
        Calculates euclidean distance

        Arguments:
            data {list} -- list of sentences.

        Returns:
            np.array -- similarity matrix of input sentences.
        """        
        return self.__common_routine(data, self.distances['euclidean'])

    def chebyshev_distance(self, data):
        """
        Calculates chebyshev distance

        Arguments:
            data {list} -- list of sentences.

        Returns:
            np.array -- similarity matrix of input sentences.
        """        
        return self.__common_routine(data, self.distances['chebyshev'])

    def canberra_distance(self, data):
        """
        Calculates canberra distance

        Arguments:
            data {list} -- list of sentences.

        Returns:
            np.array -- similarity matrix of input sentences.
        """        
        return self.__common_routine(data, self.distances['canberra'])

    def braycurtis_distance(self, data):
        """
        Calculates braycurtis distance

        Arguments:
            data {list} -- list of sentences.

        Returns:
            np.array -- similarity matrix of input sentences.
        """        
        return self.__common_routine(data, self.distances['braycurtis'])

    # TODO: This is not needed anymore.
    def __common_routine(self, data, distance_method):
        # TODO: Reomove this line below
        data_reshaped = [tf.squeeze(x).numpy() for x in data]
        # data_reshaped = [x.tolist() for x in data]
        distances = np.array([distance_method(x, y)
                              for x in data_reshaped for y in data_reshaped])
        similarities = 1 - minmax_scale(distances)
        similarity_matrix = similarities.reshape(len(data), len(data))
        return similarity_matrix
