import tensorflow as tf
import tensorflow_hub as hub
import bert
import h5py
FullTokenizer = bert.bert_tokenization.FullTokenizer

""" 
Adapted and Modifided From https://colab.research.google.com/drive/1hMLd5-r82FrnFnBub-B-fVW78Px4KPX1#scrollTo=gmR3jHYE_y3X
A wrapper class around the main functionality of BERT
The primary method is get_embeddings
"""


class BertModel:

    '''
    Set Up Requried Objects For BERT
    '''

    def __init__(self,
                 max_seq_length=128,
                 URL='https://tfhub.dev/tensorflow/bert_en_cased_L-24_H-1024_A-16/2',
                 trainable=False,
                 delete_models=True):

        self.max_seq_length = max_seq_length
        self.url = URL
        self.trainable = trainable
        self.delete_models = delete_models
        self.__is_init = False

    '''
  Takes a list of sentences and returns their embeddings.
  Returns the average of the sequence embeddings
  '''

    def __call__(self, sentences):
        self.init_model() if not self.__is_init else print('Already Initialized')

        input_words_ids, input_masks, segment_ids = self.__tokenize_sentences(
            sentences)
        __, sequence_output = self.model(
            [input_words_ids, input_masks, segment_ids])

        if self.delete_models:
            del self.model

        return tf.math.reduce_mean(sequence_output, axis=1)

    def init_model(self):
        print('Initializing')
        self.model = hub.KerasLayer(self.url, trainable=self.trainable)

        self.tokenizer = self.get_tokenizer()

        self.__is_init = True

    def get_tokenizer(self):
        vocab_file = self.model.resolved_object.vocab_file.asset_path.numpy()
        do_lower_case = self.model.resolved_object.do_lower_case.numpy()
        return FullTokenizer(vocab_file, do_lower_case)

    def __tokenize_sentences(self, sentences):
        input_words_ids = []
        input_masks = []
        segments_ids = []

        for sentence in sentences:
            input_word_ids, input_mask, segment_ids = self.__tokenize_sentence(
                sentence)
            input_words_ids.append(input_word_ids)
            input_masks.append(input_mask)
            segments_ids.append(segment_ids)

        return input_words_ids, input_masks, segments_ids

    def __tokenize_sentence(self, sentence):
        stokens = self.tokenizer.tokenize(sentence)
        stokens = ["[CLS]"] + stokens + ["[SEP]"]

        input_ids = self.__get_ids(stokens)
        input_masks = self.__get_masks(stokens)
        input_segments = self.__get_segments(stokens)

        return input_ids, input_masks, input_segments

    def __get_masks(self, tokens):
        """Mask for padding"""
        if len(tokens) > self.max_seq_length:
            raise IndexError("Token length more than max seq length!")
        return [1]*len(tokens) + [0] * (self.max_seq_length - len(tokens))

    def __get_segments(self, tokens):
        """Segments: 0 for the first sequence, 1 for the second"""
        if len(tokens) > self.max_seq_length:
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
