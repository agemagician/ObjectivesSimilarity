import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
import time

from modules.models.BertModel import BertModel
from modules.models.AlbertModel import AlbertModel
from modules.models.ElmoModel import ElmoModel
from modules.models.USEModel import USEModel
from modules.models.Word2VecModel import Word2VecModel
from modules.models.NNLMModel import NNLMModel
from modules.ext.utils import assert_can_write_hd5


class EmbeddingsGenerator:
    """
    Handles the process of generating word embeddings for various modules. Reads from file or from direct input.
    """

    def __init__(self, elmo_url='https://tfhub.dev/google/elmo/3',
                 use_url='https://tfhub.dev/google/universal-sentence-encoder/4',
                 word2vec_url='https://tfhub.dev/google/Wiki-words-500/2',
                 nnlm_url='https://tfhub.dev/google/nnlm-en-dim128/2',
                 bert_url='https://tfhub.dev/tensorflow/bert_en_cased_L-12_H-768_A-12/2',
                 albert_url='https://tfhub.dev/tensorflow/albert_en_xxlarge/1',
                 delete_models=True,
                 verbose = True):
        """[summary]

        Keyword Arguments:
            elmo_url {str} -- url for the elmo model from tfhub. (default: {'https://tfhub.dev/google/elmo/3'})
            use_url {str} -- url for the use model from tfhub. (default: {'https://tfhub.dev/google/universal-sentence-encoder/4'})
            word2vec_url {str} -- url for the word2vec model from tfhub. (default: {'https://tfhub.dev/google/Wiki-words-500/2'})
            nnlm_url {str} -- url for the nnlm model from tfhub. (default: {'https://tfhub.dev/google/nnlm-en-dim128/2'})
            bert_url {str} -- url for the bert model from tfhub. (default: {'https://tfhub.dev/tensorflow/bert_en_cased_L-12_H-768_A-12/2'})
            albert_url {str} -- url for the albert model from tfhub. (default: {'https://tfhub.dev/tensorflow/albert_en_xxlarge/1'})
            delete_models {bool} -- whether to delete the downloaded models after generating the embeddings or to keep them stored.(default: {True})
        """

        self.elmo_url = elmo_url
        self.use_url = use_url
        self.word2vec_url = word2vec_url
        self.nnlm_url = nnlm_url
        self.bert_url = bert_url
        self.albert_url = albert_url

        self.delete_models = delete_models
        self.verbose = verbose
        
        self.models = {
            'use': USEModel,
            'word2vec': Word2VecModel,
            'nnlm': NNLMModel,
            'bert': BertModel,
            'albert': AlbertModel,
            'elmo': ElmoModel
        }

        self.supported_model_names = self.models.keys()



    def get_supported_model_names(self):
        """
        Returns a list of the supported word embedding model names by the class.

        Returns: [list]
        """
        return self.supported_model_names


    def save_embeddings_for_csv(self, input_csv, output_dir, output_prefix='', chunk_size=100, model_names=None, id_key= 'id', sentence_key= 'sentence'):
        """
        Reads input sentences from a csv file, generates their embeddings and stores them in hd5 files.

        Arguments:
            input_csv {str} -- path to the csv file with the sentences
            output_dir {str} -- directory where the hd5 files will be created

        Keyword Arguments:
            output_prefix {str} -- an optional custom prefix to add to the created files (default: {''})
            chunk_size {int} -- the function reads from the csv file in chunks, this variable specifies the chunk size (default: {100})
            model_names {list} -- a list of the model names to generate the embeddings for. 
                                If None, it will be set to self.supported_model_names (default: {None})
            id_key {str} -- name of the column in the csv file with the sentence ids (default: {id})
            sentence_key {str} -- name of the column in the csv file with the sentence values (default: '{sentence})

        Returns:
            dict --  a dictionary of the form {model_name: output hdf5 file location}
        """
        self.delete_models = False
        model_names = model_names if model_names is not None else self.supported_model_names

        created_files = {}
        for model_name in model_names:
            store_name = '{}/{}_{}_embeddings.h5'.format(
                output_dir, output_prefix, model_name)
            self.__save_embeddings_for_model(
                model_name, store_name, input_csv, chunk_size, id_key, sentence_key)
            created_files[model_name] = store_name

        return created_files


    def __print(self, text):
        print(text) if self.verbose else None

    def __save_embeddings_for_model(self, model_name, store_name, input_csv, chunk_size, id_key, sentence_key):
        """
        A helper used by save_embeddings_for_csv. Handles writing the embeddings for a single model.

        Arguments:
            model_name {str} -- model to write the embeddings for.  
            store_name {str} -- path to the hd5 store to write the embeddings in.
            input_csv {str} -- path to the csv file to read the sentences from.
            chunk_size {int} -- the function reads from the csv file in chunks, this variable specifies the chunk size.
            id_key {str} -- name of the column in the csv file with the sentence ids
            sentence_key {str} -- name of the column in the csv file with the sentence values
        """

        assert_can_write_hd5(store_name)
        self.__print('Embeddings Generator: Creating {} embeddings..'.format(model_name))
        store = pd.HDFStore(store_name)

        for i, chunk in enumerate(pd.read_csv(input_csv, chunksize=chunk_size)):
            self.__print('Embeddings Generator: Reading chunk_{} sentences from {}'.format(i, input_csv))
            sentences = chunk[sentence_key]
            embeds = self.__get_embeddings(model_name, sentences)

            s = pd.DataFrame(embeds.numpy())
            s[id_key] = chunk[id_key].values
            s = s.set_index(id_key)

            store.append('chunk_{}'.format(i), s)

        store.close()
        self.__print('Embeddings Generator: Done Writing {} embeddings to {}\n'.format(model_name, store_name))
        self.__delete_model(model_name) if self.delete_models else None

    def get_embeddings(self, model_name, inputs):
        """
        generates and returns embeddings of a given model for a given list of sentences

        Arguments:
            model_name {str} -- name of the model to generate the embeddings with, must be from the supported list of model names.
            inputs {list} -- list of sentences to generate the embeddings for.

        Returns:
            list -- list of generated embeddings
        """
        embeddings = self.__get_embeddings(model_name, inputs)
        print('################# ', type(embeddings))
        self.__delete_model(model_name) if self.delete_models else None

        return embeddings

    def __get_embeddings(self, model_name, inputs):
        """
        Helper used by both get_embeddings, and __save_embeddings_for_model. 
        Checks if model name is supported. Loads the model from tfhub if not loaded.
        Calls the model and returns the embeddings for the given inputs.

        Arguments:
            model_name {str} -- name of the model to generate the embeddings with, must be from the supported list of model names.
            inputs {list} -- list of sentences to generate the embeddings for.

        Returns:
            list -- list of generated embeddings
        """
        if model_name in self.supported_model_names:
            model = self.__load_or_get_model(model_name)
            return model(inputs)
        else:
            self.__print('Embeddings Generator: Unsupported model name "{}"'.format(model_name))

    def __model_loader(self, model_name):
        """
        Helper used by __get_embeddings. Selects the correct model based on the input model_name

        Arguments:
            model_name {str} --  name of the model to generate the embeddings with, must be from the supported list of model names.

        Returns:
            hub.load, BertModel, or AlbertModel --  an object which returns the embedding on __call__
        """
        model_url = getattr(self, model_name + '_url')
        modelObject = self.models[model_name]
        return modelObject(URL=model_url, delete_models=self.delete_models)

    def __load_or_get_model(self, model_name):
        """
        Helper used by __model_loader. Loads the model if not loaded before.

        Arguments:
            model_name {str} -- model_name

        Returns:
            hub.load, BertModel, or AlbertModel --  an object which returns the embedding on __call__
        """
        if not hasattr(self, model_name):
            self.__print('Embeddings Generator: Loading {} pretrained model'.format(model_name))
            setattr(self, model_name, self.__model_loader(model_name))
        else:
            self.__print('Embeddings Generator: {} already loaded'.format(model_name))

        return getattr(self, model_name)

    def __delete_model(self, model_name):
        """
        Deletes the model attribute from this instance.

        Arguments:
            model_name {str} -- model name to be deleted
        """
        self.__print('Embeddings Generator: Unloading {} pretrained model'.format(model_name))
        delattr(self, model_name)

    def print_shapes_and_time(self, sentences):
        """
        Generates embeddings for input sentences. prints their shape and the time taken to generate them

        Arguments:
            sentences {list} -- list of sentences to generate the embeddings for.
        """
        for model_name in self.supported_model_names:
            start = time.process_time()
            embeddings = self.get_embeddings(model_name, sentences)
            print(model_name, time.process_time() - start, embeddings.shape)
