from sentence_transformers import SentenceTransformer
import torch

class BaseSentenceTransformer:

    '''
    Set Up Requried Objects For a gpt Model From Hugging Face
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

        embeddings = self.__embed(sentences)

        if self.delete_models:
            del self.model
            self.__is_init = False

        return embeddings.detach()

    def init_model(self):
        print('{}: Initializing'.format(self.model_name))
        self.model = SentenceTransformer(self.url)
        self.__is_init = True

    def __embed(self, sentences):
        sentence_embeddings = self.model.encode(sentences)
        sentence_embeddings = torch.tensor(sentence_embeddings)

        return sentence_embeddings