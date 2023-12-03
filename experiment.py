
from models.DumbModel import DumbModel
from models.LR_Model import LR_Model
import os

class Experiment(object):
    def __init__(self, config) -> None:
        self.config = config
        self.model = get_model(config)     
        
        
    def train(self):
        print("training ...")
        self.model.train()
    
    def test(self):
        self.model.test()
        
def get_model(config):
    if config['model_type'] == "dumb":
        return DumbModel(config)
    if config['model_type'] == "LR":
        return LR_Model(config)
    # if config['model_type'] == "dumb":
    #     return DumbModel(config)
    # if 