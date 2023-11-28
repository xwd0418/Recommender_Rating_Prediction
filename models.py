from tqdm import tqdm
import numpy as np


def get_model(config):
    if config['model_type'] == "dumb":
        return DumbModel(config)
    

class Model(object):
    """abstraction. 
    Each model should be able to train and predict
    Note that train() takes all training data, but predict only takes one datum

    Args:
        ABC (_type_): _description_
    """
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
    
    # abstractmethod
    def predict(self, datum):
        pass
    
    # abstractmethod
    def iterate(self):
        """update the model once"""
        pass
    
    # abstractmethod
    def save_best_params(self):
        """save best params either to memory or to disk"""
        pass
    
    # abstractmethod
    def load_best_params(self):
        """load best params either from memory or from disk"""
        pass
    
    
    def train(self, train_X, train_Y, val_X, val_Y):
        """update model. 
        This function is only useful when models will keep updating and 
        numerically converge to some point 

        Args:
            train_X (): train data
            train_Y (): train labels
            val_X: val data
            val_Y: val label
        """
        
        self.train_X, self.train_Y, self.val_X, self.val_Y = train_X, train_Y, val_X, val_Y
        pbar = tqdm(range(self.config['epoch']))
        epoch_of_converge = 0
        self.smallest_val_MSE = float('inf')
        file_train_output = open(self.config['file_output_dir_path']+"training_log.txt", "w") 
        
        for epoch in pbar:
            if epoch_of_converge > 10:
                break
            # print("now interate")
            self.iterate()
            # print("now")
            curr_val_MSE = self.compute_val_loss(self.val_X, self.val_Y)
            if curr_val_MSE < self.smallest_val_MSE:
                self.smallest_val_MSE = curr_val_MSE
                epoch_of_converge = 0
                self.save_best_params()
            else:
                epoch_of_converge += 1
            
            # uncomment this if you want a progress bar    
            # pbar.set_description(f"curr_MSE: {curr_val_MSE}")
            file_train_output.write(f"At epoch {epoch}, validation MSE: {curr_val_MSE}/n")
            
    def compute_loss(self, X, Y):
        ground_truth = Y
        pred =  np.array([ self.predict(datum)  for datum in X])
        return  ((ground_truth - pred)**2).mean(axis=0)
    
class DumbModel(Model):
    '''
    this dumb model does nothing but directly gives a rating
    '''
    def train(self, train_X, train_Y):
        pass
    
    def predict(self, config):
        return config['dumb_output']