from transformers import RobertaTokenizer, RobertaModel
import torch

class Roberta_Model:

    '''
    Set Up Requried Objects For a gpt Model From Hugging Face
    '''

    def __init__(self, model_name='Roberta', URL='roberta-base', delete_models=True):

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

        embeddings = [self.__embed(sentence) for sentence in sentences]
        embeddings = torch.stack(embeddings)

        if self.delete_models:
            del self.model
            del self.tokenizer
            self.__is_init = False

        return embeddings.detach()

    def init_model(self):
        print('{}: Initializing'.format(self.model_name))
        self.tokenizer = RobertaTokenizer.from_pretrained(self.url)
        self.model = RobertaModel.from_pretrained(self.url)
        self.__is_init = True

    def __embed(self, sentence):
        inputs_ids = self.tokenizer.encode(sentence, return_tensors='pt', max_length=512)
        outputs = self.model(inputs_ids)
        embeddings = outputs[0][0].mean(0)

        return embeddings