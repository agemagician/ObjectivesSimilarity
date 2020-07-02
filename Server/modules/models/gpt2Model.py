from transformers import GPT2Tokenizer, GPT2Model
import torch

class gpt2Model:

    '''
    Set Up Requried Objects For a gpt2 Model From Hugging Face
    '''

    def __init__(self, model_name='gpt2', URL='gpt2', delete_models=True):

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
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.url)
        self.model = GPT2Model.from_pretrained(self.url)
        self.__is_init = True

    def __embed(self, sentence):
        input_ids = self.tokenizer(sentence, return_tensors="pt")
        outputs = self.model(**input_ids)

        embeddings = outputs[0][0].mean(0)

        return embeddings