import tokenization
from BertModel import BertModel

"""
A wrapper class around Albert
"""
class AlbertModel(BertModel):
  def __init__(self, max_seq_length = 128, URL= 'https://tfhub.dev/tensorflow/albert_en_base/1', trainable = False, delete_models = True):
    super(AlbertModel, self).__init__(max_seq_length, URL, trainable, delete_models)

  def get_tokenizer(self):
    sp_model_file = self.model.resolved_object.sp_model_file.asset_path.numpy()
    return tokenization.FullSentencePieceTokenizer(sp_model_file)