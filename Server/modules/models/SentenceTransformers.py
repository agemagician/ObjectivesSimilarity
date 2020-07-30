from modules.models.BaseSentenceTransformer import BaseSentenceTransformer

class SentenceBert(BaseSentenceTransformer):

      def __init__(self, model_name='sentence_bert', URL='bert-large-nli-stsb-mean-tokens', delete_models=True):
            super(SentenceBert, self).__init__(model_name, URL, delete_models)


class SentenceRoBerta(BaseSentenceTransformer):

      def __init__(self, model_name='sentence_roberta', URL='roberta-large-nli-stsb-mean-tokens', delete_models=True):
            super(SentenceRoBerta, self).__init__(model_name, URL, delete_models)


class SentenceDistilBert(BaseSentenceTransformer):

      def __init__(self, model_name='sentence_distilbert', URL='distilbert-base-nli-stsb-mean-tokens', delete_models=True):
            super(SentenceDistilBert, self).__init__(model_name, URL, delete_models)