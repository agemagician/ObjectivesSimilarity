import numpy as np
import tensorflow as tf
import pandas as pd

from scipy.spatial import distance
from sklearn.preprocessing import minmax_scale
from utils import assert_row_dim, order_keys, get_non_dupicate_keys, update_heaps
from keyValueHeap import keyValueHeap


class SimilarityCalculator:

    def __init__(self):

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



    def __write_distances_results(self, distances_result, ready_keys):
        for distance_metric, distance_results in distances_result.items():
            store_name = '{}/{}_index.h5'.format(self.output_dir, distance_metric)

            print('Writing To {}'.format(store_name), end = '... ')
            store = pd.HDFStore(store_name)

            for key, results in distance_results.items():
                if key not in ready_keys: continue
                ordered_results_df = pd.DataFrame(keyValueHeap.reverse(results), columns = ['id', 'distance'])

                store.append('key_{}'.format(key), ordered_results_df)

            store.close()
            print('Done Writing {} data'.format(distance_metric))


    def from_hdf(self, file_path):
        print('Opening ', file_path)
        hdf = pd.HDFStore(file_path, mode='r')

        try:
            for chunk1_key in hdf.keys():
                chunk1 = hdf.get(chunk1_key)
                
                distances_results = dict()
                print('\nStarting Chunk {}'.format(chunk1_key))
                for chunk2_key in hdf.keys():
                    chunk2 = hdf.get(
                        chunk2_key) if chunk2_key != chunk1_key else chunk1
                    self.__calculate_chunk_similarity(
                        chunk1, chunk2, distances_results)

                self.__write_distances_results(distances_results, chunk1.index)
                print('Done with Chunk {}\n'.format(chunk1_key))
        finally:
            hdf.close()
            print('Closing ',file_path)

    def __calculate_chunk_similarity(self, chunk1, chunk2, distances_results):
        chunk1_row_keys, chunk2_row_keys = chunk1.index, chunk2.index
        keys_to_compare = get_non_dupicate_keys(
            chunk1_row_keys, chunk2_row_keys)

        for row1_key, row2_key in keys_to_compare:
            for distance_metric, distance_function in self.distances.items():

                row1 = chunk1.loc[row1_key]
                assert_row_dim(row1.ndim, row1_key)

                row2 = chunk2.loc[row2_key]
                assert_row_dim(row2.ndim, row2_key)

                similarity = (-1) * distance_function(row1, row2)
                update_heaps(distances_results, distance_metric,
                             row1_key, row2_key, similarity)

    def cosine(self, data):
        return self.__common_routine(data, self.distances['cosine'])

    def minkowski(self, data):
        return self.__common_routine(data, self.distances['minkowski'])

    def correlation(self, data):
        return self.__common_routine(data, self.distances['correlation'])

    # Does not work if normalize is False
    def jensen_shannon(self, data, normalize=True):
        return self.__common_routine(data, self.distances['jensenshannon'])

    def cityblock_distance(self, data):
        return self.__common_routine(data, self.distances['cityblock'])

    def squared_euclidian_distance(self, data):
        return self.__common_routine(data, self.distances['sqeuclidean'])

    def euclidean_distance(self, data):
        return self.__common_routine(data, self.distances['euclidean'])

    def chebyshev_distance(self, data):
        return self.__common_routine(data, self.distances['chebyshev'])

    def canberra_distance(self, data):
        return self.__common_routine(data, self.distances['canberra'])

    def braycurtis_distance(self, data):
        return self.__common_routine(data, self.distances['braycurtis'])

    # TODO: Check This Still Works
    def __common_routine(self, data, distance_method):
        # TODO: Reomove this line below
        data_reshaped = [tf.squeeze(x).numpy() for x in data]
        # data_reshaped = [x.tolist() for x in data]
        distances = np.array([distance_method(x, y)
                              for x in data_reshaped for y in data_reshaped])
        similarities = 1 - minmax_scale(distances)
        similarity_matrix = similarities.reshape(len(data), len(data))
        return similarity_matrix
