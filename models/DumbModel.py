from models.abstract_model import Model
class DumbModel(Model):
    '''
    this dumb model does nothing but directly gives a rating
    '''
    def predict(self, datum):
        return self.config['dumb_output']
    
    def train(self):
        pass
    
    def load_best_params(self):
        return self