import tensorflow_hub as hub

class BasicTFHubModel:

    '''
    Set Up Requried Objects For a Basic TF HUb Model
    '''

    def __init__(self, model_name, URL, delete_models=True):

        self.url = URL
        self.model_name = model_name
        self.delete_models = delete_models
        self.__is_init = False

    '''
  Takes a list of sentences and returns their embeddings.
  Returns the average of the sequence embeddings
  '''

    def __call__(self, sentences):
        self.init_model() if not self.__is_init else print('{}: Already Initialized'.format(self.model_name))

        embeddings = self.model(sentences)

        if self.delete_models:
            del self.model
            self.__is_init = False

        return embeddings

    def init_model(self):
        print('{}: Initializing'.format(self.model_name))
        self.model = hub.load(self.url)
        self.__is_init = True