import string
import nltk

from nltk.corpus import stopwords 
from collections import Counter
from nltk import ngrams
from nltk.tokenize import sent_tokenize, word_tokenize
from operator import itemgetter   

nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words('english')) 

def common_ngrams(sentences, ngram_range=(2,4), n_common=10):
    sentences = ' '.join(sentences)
    sentence = sentences.translate(str.maketrans('','', string.punctuation))
    token_words = word_tokenize(sentence)
    token_words = [w for w in token_words if not w in stop_words] 

    cmns = []
    for ngram in range(*ngram_range):
        ngram_counts = Counter(ngrams(token_words, ngram))
        cmns.extend(ngram_counts.most_common(n_common))

    # print('common n-grams')
    # print(cmns[:n_common])

    cmns.sort(key=itemgetter(1), reverse=True)
    return [cmn_i for cmn in cmns[:n_common] for cmn_i in cmn[0]]
 
def tokenize(sentences):
    sentences = ' '.join(sentences)
    sentence = sentences.translate(str.maketrans('','', string.punctuation))
    token_words = word_tokenize(sentence)
    token_words = [w for w in token_words if not w in stop_words] 
    return token_words

def common_ngram_txt(base, results, size=2):
    tokens1 = tokenize([base])
    tokens2 = tokenize(results)

    ng1=set(ngrams(tokens1, size))
    ng2=set(ngrams(tokens2, size))
 
    print(ng1)
    print(ng2)

    match=set.intersection(ng1,ng2)
    print('..found {}'.format(len(match)))
 
    print(match)

    match = list(match)
    match = [m for mm in match for m in mm]
    return match