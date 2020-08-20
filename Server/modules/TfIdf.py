from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd

class TfIdf():

  def __init__(self):    
    self.tfidf = TfidfVectorizer(stop_words='english', ngram_range=(2,5))

  
  def fit(self, data_file, objective_key):
    data = pd.read_csv(data_file)
    self.tfidf.fit_transform(data[objective_key])
      
  def get_keys(self, targets, n=20):
    response = self.tfidf.transform(targets)

    feature_array = np.array(self.tfidf.get_feature_names())
    tfidf_sorting = np.argsort(response.toarray()).flatten()[::-1]

    top_n = feature_array[tfidf_sorting][:n]

    # print('Top {} Important Words'.format(n))
    # print(top_n)

    expanded_list = [i for item in top_n.tolist() for i in item.split()]
    return expanded_list