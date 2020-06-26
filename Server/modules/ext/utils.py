import os
import shutil
import sys
from modules.ext.keyValueHeap import keyValueHeap


def assert_can_write_hd5(file_name):
    if os.path.exists(file_name):
        try: 
            os.remove(file_name)
        except OSError as e:
            raise Exception('Error while removeing file: {} \n\n Message:{}.'.format(file_name, e))


def assert_row_dim(row_dim, row_key):
    if row_dim > 1:
        raise Exception(
            'ID Error: Multiple Entries With The same ID = {}'.format(row_key))


def order_keys(key1, key2):
    return (key1, key2) if key1 < key2 else (key2, key1)


def format_key(key1, key2):
    return '{}_{}'.format(*order_keys(key1, key2))


def get_non_dupicate_keys(keys_1, keys_2):
    keys = []
    seen = set()
    for key_1 in keys_1:
        for key_2 in keys_2:
            ordered_key = order_keys(key_1, key_2)
            if ordered_key not in seen and key_1 != key_2:
                keys.append((key_1, key_2))
                seen.add(ordered_key)

    return list(keys)
    
def update_heaps(heaps, distance_metric, key1, key2, distance, MAX):
    distance_dict = heaps.setdefault(distance_metric, {})

    key1_heap = distance_dict.setdefault(key1, keyValueHeap(MAX))
    key1_heap.push(distance, key2)

    key2_heap = distance_dict.setdefault(key2, keyValueHeap(MAX))
    key2_heap.push(distance, key1)
   
