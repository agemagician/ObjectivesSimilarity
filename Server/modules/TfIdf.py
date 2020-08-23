from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd

class TfIdf():

  def __init__(self):    
    self.tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1,3))

  
  def fit(self, data_file, objective_key):
    data = pd.read_csv(data_file)
    self.tfidf.fit_transform(data[objective_key])
      
  def get_keys(self, targets, n=20):
    response = self.tfidf.transform(targets)

    feature_array = np.array(self.tfidf.get_feature_names())
    tfidf_sorting = np.argsort(response.toarray()).flatten()[::-1]

    top_n = feature_array[tfidf_sorting][:n]

    expanded_list = top_n.tolist()
    return expanded_list