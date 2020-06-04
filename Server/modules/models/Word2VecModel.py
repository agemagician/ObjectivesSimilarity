from modules.models.BasicTFHubModel import BasicTFHubModel

class Word2VecModel(BasicTFHubModel):

    '''
    Set Up Requried Objects For Word2Vec
    '''

    def __init__(self, 
                 URL='https://tfhub.dev/google/Wiki-words-500/2',
                 delete_models=True):

        super(Word2VecModel, self).__init__(
            model_name='Word2Vec',
            URL=URL,
            delete_models=delete_models)