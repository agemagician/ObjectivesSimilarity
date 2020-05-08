""" 
Adapted and Modifided From https://colab.research.google.com/drive/1hMLd5-r82FrnFnBub-B-fVW78Px4KPX1#scrollTo=gmR3jHYE_y3X
A wrapper class around the main functionality of BERT
The primary method is get_embeddings
"""

class BertModel:
  '''
  Set Up Requried Objects For BERT
  '''
  def __init__(self, bert_layer):
    self.max_seq_length = 128
    input_word_ids = tf.keras.layers.Input(shape=(self.max_seq_length,), dtype=tf.int32,
                                          name="input_word_ids")
    input_mask = tf.keras.layers.Input(shape=(self.max_seq_length,), dtype=tf.int32,
                                      name="input_mask")
    segment_ids = tf.keras.layers.Input(shape=(self.max_seq_length,), dtype=tf.int32,
                                        name="segment_ids")
    
    pooled_output, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])
    self.model = Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=[pooled_output, sequence_output])

    vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
    do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
    self.tokenizer = FullTokenizer(vocab_file, do_lower_case)

  '''
  Takes a list of sentences and returns their embeddings
  '''
  def get_embeddings(self, sentences):
    return np.array([self.__embed_single_sentence(sentence) for sentence in sentences]).reshape(len(sentences),-1)


  def __embed_single_sentence(self, sentence):
    stokens = self.tokenizer.tokenize(sentence)
    stokens = ["[CLS]"] + stokens + ["[SEP]"]

    input_ids = self.__get_ids(stokens)
    input_masks = self.__get_masks(stokens)
    input_segments = self.__get_segments(stokens)

    pool_embs, all_embs = self.model.predict([[input_ids],[input_masks],[input_segments]])

    return pool_embs

  def __get_masks(self, tokens):
    """Mask for padding"""
    if len(tokens)> self.max_seq_length:
        raise IndexError("Token length more than max seq length!")
    return [1]*len(tokens) + [0] * (self.max_seq_length - len(tokens))


  def __get_segments(self, tokens):
      """Segments: 0 for the first sequence, 1 for the second"""
      if len(tokens)> self.max_seq_length:
          raise IndexError("Token length more than max seq length!")
      segments = []
      current_segment_id = 0
      for token in tokens:
          segments.append(current_segment_id)
          if token == "[SEP]":
              current_segment_id = 1
      return segments + [0] * (self.max_seq_length - len(tokens))


  def __get_ids(self, tokens):
      """Token ids from Tokenizer vocab"""
      token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
      input_ids = token_ids + [0] * (self.max_seq_length-len(token_ids))
      return input_ids