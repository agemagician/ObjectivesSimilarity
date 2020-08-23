from transformers import T5Tokenizer, T5Model
import torch

class T5_Model:

    '''
    Set Up Requried Objects For a gpt Model From Hugging Face
    '''

    def __init__(self, model_name='T5', URL='t5-small', delete_models=True):

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

        return embeddings.cpu()

    def init_model(self):
        print('{}: Initializing'.format(self.model_name))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = T5Tokenizer.from_pretrained(self.url)
        self.model = T5Model.from_pretrained(self.url).to(self.device)
        self.model.eval()
        self.__is_init = True

    def __embed(self, sentence):
        with torch.no_grad():

            inputs_ids = self.tokenizer.encode(sentence, return_tensors='pt').to(self.device)
            outputs = self.model(input_ids=inputs_ids, decoder_input_ids=inputs_ids)

            embeddings = outputs[0][0].mean(0)

            return embeddings