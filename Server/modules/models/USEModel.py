from modules.models.BasicTFHubModel import BasicTFHubModel

class USEModel(BasicTFHubModel):

    '''
    Set Up Requried Objects For Word2Vec
    '''

    def __init__(self, 
                 URL='https://tfhub.dev/google/universal-sentence-encoder/4',
                 delete_models=True):

        super(USEModel, self).__init__(
            model_name='USE',
            URL=URL,
            delete_models=delete_models)