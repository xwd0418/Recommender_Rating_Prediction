from dataset import dataset_split
from models import get_model
import os

class Experiment(object):
    def __init__(self, config) -> None:
        self.config = config
        
        # load dataset
        self.train_X, self.train_Y, \
        self.val_X, self.val_Y, \
        self.test_X, self.test_Y = dataset_split()
        
        # load model
        self.model = get_model(config)
        
        
        
    def train(self):
        self.model.train(self.train_X, self.train_Y, self.val_X, self.val_Y)
    
    def test(self):
        file_test_output = open(self.config['file_output_dir_path']+"testing_log.txt", "w") 
        best_model = self.model.load_best_params()
        
        test_mse = best_model.compute_val_loss(self.test_X, self.test_Y)
        file_test_output.write(f"Test MSE is {test_mse}")
        