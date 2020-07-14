import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer
from collections import defaultdict
from nltk.corpus import stopwords 
import string

nltk.download('punkt')
nltk.download('stopwords')

porter = PorterStemmer()
stem_word_map = defaultdict(set)
stop_words = set(stopwords.words('english')) 


def stemSentence(sentence, stem_word_map):
    sentence = sentence.translate(str.maketrans('','', string.punctuation))
    token_words=word_tokenize(sentence)
    token_words = [w for w in token_words if not w in stop_words] 

    token_words
    unique_stems = set()
    for word in token_words:
        stem = porter.stem(word)
        stem_word_map[stem].add(word)
        unique_stems.add(stem)

    return unique_stems, stem_word_map
  

def get_common_words(objectives):
    print('oBjectives are :::::::::::::::::::::::::::::::::::::::::::::::: ', objectives)
    stem_word_map = defaultdict(set)
    all_unique_stems = []
    for objective in objectives:
        unique_stems, stem_word_map = stemSentence(objective, stem_word_map)
        all_unique_stems.append(unique_stems)

    common_stems = all_unique_stems[0]
    for unique_stems in all_unique_stems[0:]:
        common_stems = common_stems.intersection(unique_stems)

    common_words = []
    for stem in common_stems:
        common_words.extend(stem_word_map[stem])

    return common_words