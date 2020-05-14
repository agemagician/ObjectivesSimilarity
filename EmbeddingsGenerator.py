import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
import time

from BertModel import BertModel
from AlbertModel import AlbertModel
from utils import assert_can_write_hd5

class EmbeddingsGenerator:
    def __init__(self, elmo_url='https://tfhub.dev/google/elmo/3',
                 use_url='https://tfhub.dev/google/universal-sentence-encoder/4',
                 word2vec_url='https://tfhub.dev/google/Wiki-words-500/2',
                 nnlm_url='https://tfhub.dev/google/nnlm-en-dim128/2',
                 bert_url='https://tfhub.dev/tensorflow/bert_en_cased_L-12_H-768_A-12/2',
                 albert_url='https://tfhub.dev/tensorflow/albert_en_xxlarge/1',
                 delete_models=True):

        self.elmo_url = elmo_url
        self.use_url = use_url
        self.word2vec_url = word2vec_url
        self.nnlm_url = nnlm_url
        self.bert_url = bert_url
        self.albert_url = albert_url

        self.delete_models = delete_models
        self.supported_model_names = [
            'elmo', 'use', 'word2vec', 'nnlm', 'bert', 'albert']

    def get_embeddings_for_csv(self, input_file, output_dir, output_file = '', chunk_size=100, model_names=None):
        self.delete_models = False
        model_names = model_names if model_names is not None else self.supported_model_names

        for model_name in model_names:
            store_name = '{}/{}_{}_embeddings.h5'.format(output_dir, output_file, model_name)
            assert_can_write_hd5(store_name)

            print('Writing To ', store_name)
            store = pd.HDFStore(store_name)

            for i, chunk in enumerate(pd.read_csv(input_file, chunksize=chunk_size)):
                sentences = chunk['sentence']
                embeds = self.__get_embeddings(model_name, sentences)

                s = pd.DataFrame(embeds.numpy())
                s['id'] = chunk['id'].values
                s = s.set_index('id')

                store.append('chunk_{}'.format(i), s)

            store.close()
            print('Done Writing {} embeddings'.format(model_name))
            self.__delete_model(model_name) if self.delete_models else None

    def get_embeddings(self, model_name, inputs):
        embeddings = self.__get_embeddings(model_name, inputs)
        self.__delete_model(model_name) if self.delete_models else None

        return embeddings

    def __get_embeddings(self, model_name, inputs):
        if model_name in self.supported_model_names:
            model = self.__load_or_get_model(model_name)
            if model_name == 'elmo':
                return model.signatures['default'](tf.constant(inputs))["default"]
            else:
                return model(inputs)
        else:
            print('Unsupported Model: ', model_name)

    def __model_loader(self, model_name):
        model_url = getattr(self, model_name + '_url')

        if model_name in ['elmo', 'use', 'word2vec', 'nnlm']:
            return hub.load(model_url)

        elif model_name == 'bert':
            return BertModel(URL=model_url, delete_models=self.delete_models)

        elif model_name == 'albert':
            return AlbertModel(URL=model_url, delete_models=self.delete_models)

    def __load_or_get_model(self, model_name):
        if not hasattr(self, model_name):
            print(model_name, ": Loading")
            setattr(self, model_name, self.__model_loader(model_name))
        else:
            print(model_name, " Already Loaded")

        return getattr(self, model_name)

    def __delete_model(self, model_name):
        delattr(self, model_name)

    def print_shapes_and_time(self, sentences):
        for model_name in self.supported_model_names:
            start = time.process_time()
            embeddings = self.get_embeddings(model_name, sentences)
            print(model_name, time.process_time() - start, embeddings.shape)
