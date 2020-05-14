
from scipy.spatial import distance
from sklearn.preprocessing import minmax_scale
from keyValueHeap import keyValueHeap

import numpy as np
import tensorflow as tf
import pandas as pd


def _order_keys(key1, key2):
    return (key1, key2) if key1 < key2 else (key2, key1)


def _format_key(key1, key2):
    return '{}_{}'.format(*_order_keys(key1, key2))


def _mark_read(key1, key2, done):
    return done.add(_format_key(key1, key2))

def assert_row_shapes(row, row_key):
    if row.shape[0] > 1 :
        raise Exception('ID Error: Multiple Entries With The same ID = {}'.format(row_key))    

def _get_non_dupicate_keys(keys_1, keys_2):
    keys = []
    seen = set()
    for key_1 in keys_1:
        for key_2 in keys_2:
            ordered_key = _order_keys(key_1, key_2)
            if ordered_key not in seen and key_1 != key_2:
                keys.append((key_1, key_2))
                seen.add(ordered_key)

    return list(keys)


MAX = 10


def update_heaps(heaps, distance_metric, key1, key2, distance):
    distance_dict = heaps.setdefault(distance_metric, {})

    key1_heap = distance_dict.setdefault(key1, keyValueHeap(MAX))
    key1_heap.push(distance, key2)

    key2_heap = distance_dict.setdefault(key2, keyValueHeap(MAX))
    key2_heap.push(distance, key1)


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

            print('Writing To ', store_name, sep = '... ')
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
                for chunk2_key in hdf.keys():
                    chunk2 = hdf.get(
                        chunk2_key) if chunk2_key != chunk1_key else chunk1
                    self.__calculate_chunk_similarity(
                        chunk1, chunk2, distances_results)

                self.__write_distances_results(distances_results, chunk1.index)
                print('For Chunk {}'.format(chunk1_key))
        finally:
            hdf.close()
            print('Closing ',file_path)

    def __calculate_chunk_similarity(self, chunk1, chunk2, distances_results):
        chunk1_row_keys, chunk2_row_keys = chunk1.index, chunk2.index
        keys_to_compare = _get_non_dupicate_keys(
            chunk1_row_keys, chunk2_row_keys)

        for row1_key, row2_key in keys_to_compare:
            for distance_metric, distance_function in self.distances.items():

                row1 = chunk1.loc[row1_key]
                assert_row_shapes(row1, row1_key)

                row2 = chunk2.loc[row2_key]
                assert_row_shapes(row2, row2_key)

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
