import tensorflow as tf
import tensorflow_hub as hub

class ElmoModel:

    '''
    Set Up Requried Objects For Elmo
    '''

    def __init__(self,
                 URL='https://tfhub.dev/tensorflow/bert_en_cased_L-24_H-1024_A-16/2',
                 delete_models=True):

        self.url = URL
        self.delete_models = delete_models
        self.__is_init = False

    '''
  Takes a list of sentences and returns their embeddings.
  Returns the average of the sequence embeddings
  '''

    def __call__(self, sentences):
        self.init_model() if not self.__is_init else print('Elmo: Already Initialized')

        embeddings = self.model.signatures['default'](tf.constant(sentences))["default"]

        if self.delete_models:
            del self.model
            self.__is_init = False

        return embeddings

    def init_model(self):
        print('Elmo: Initializing')
        self.model = hub.load(self.url)
        self.__is_init = True