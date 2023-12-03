from tqdm import tqdm
import numpy as np
from dataset import dataset_split, read_csv

    

class Model(object):
    """abstraction. 
    Each model should be able to train and predict
    Note that train() takes all training data, but predict only takes one datum

    """
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        print("reading dataset ......")
        interaction_header, interactions = read_csv('archive/RAW_interactions.csv')
        self.interations = interactions
                
        X, Y = self.extract_feature(interactions)
        self.train_X, self.train_Y, self.val_X, self.val_Y, self.test_X, self.test_Y = dataset_split(X,Y)
        
       
    
    # overridable
    def extract_feature(self, interactions):
        X = ([(d[0], d[1], d[2], d[4]) for d in interactions])
        Y = np.array([int(d[3]) for d in interactions])
        return X, Y
    
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
    
    
    def train(self):
        """update model. 
        This function is only useful when models will keep updating and 
        numerically converge to some point 

        Args:
            train_X (): train data
            train_Y (): train labels
            val_X: val data
            val_Y: val label
        """
        
        
        pbar = tqdm(range(self.config['epoch']))
        epoch_of_converge = 0
        self.best_accu = 0
        file_train_output = open(self.config['file_output_dir_path']+"training_log.txt", "w") 
        
        for epoch in pbar:
            if epoch_of_converge > 10:
                break
            # print("now interate")
            self.iterate()
            # print("now")
            val_pred = np.array([ self.predict(datum)  for datum in self.val_X])
            curr_val_MSE = self.compute_loss(val_pred, self.val_Y)
            curr_val_acuu = self.compute_accu(val_pred, self.val_Y)
            if curr_val_acuu > self.best_accu:
                self.best_accu = curr_val_acuu
                epoch_of_converge = 0
                self.save_best_params()
            else:
                epoch_of_converge += 1
            
            # uncomment this if you want a progress bar    
            # pbar.set_description(f"curr_MSE: {curr_val_MSE}")
            file_train_output.write(f"At epoch {epoch}, validation MSE: {curr_val_MSE}\n")
            file_train_output.write(f"At epoch {epoch}, validation accuracy: {curr_val_acuu}\n")
            
    def compute_loss(self, pred, Y):
        ground_truth = Y
        return  ((ground_truth - pred)**2).mean(axis=0)
    
    def compute_accu(self, pred, Y):
        return np.sum(np.round(pred) == Y) /  len(Y)
    
    def test(self):
        file_test_output = open(self.config['file_output_dir_path']+"testing_log.txt", "w") 
        best_model = self.load_best_params()
        test_pred = np.array([ best_model.predict(datum)  for datum in self.test_X])
        test_mse = self.compute_loss(test_pred, self.test_Y)
        test_acc = self.compute_accu(test_pred, self.test_Y)
        file_test_output.write(f"Test MSE is {test_mse}\n")
        file_test_output.write(f"Test accuracy is {test_acc}\n")
