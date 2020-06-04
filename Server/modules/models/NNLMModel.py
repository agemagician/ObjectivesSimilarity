from modules.models.BasicTFHubModel import BasicTFHubModel

class NNLMModel(BasicTFHubModel):

    '''
    Set Up Requried Objects For Word2Vec
    '''

    def __init__(self, 
                 URL='https://tfhub.dev/google/nnlm-en-dim128/2',
                 delete_models=True):

        super(NNLMModel, self).__init__(
            model_name='NNLM',
            URL=URL,
            delete_models=delete_models)
