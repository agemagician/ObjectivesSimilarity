from transformers import XLMTokenizer, XLMModel
import torch

class xlmModel:

    '''
    Set Up Requried Objects For a XLNET Model From Hugging Face
    '''

    def __init__(self, model_name='xlm', URL='xlm-mlm-en-2048', delete_models=True):

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
        self.tokenizer = XLMTokenizer.from_pretrained(self.url)
        self.model = XLMModel.from_pretrained(self.url)
        self.__is_init = True

    def __embed(self, sentence):
        input_ids = torch.tensor(self.tokenizer.encode(sentence, add_special_tokens=False)).unsqueeze(0)
        outputs = self.model(input_ids)

        embeddings = outputs[0][0].mean(0)

        return embeddings