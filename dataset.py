import random
import csv
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from collections import defaultdict
from datetime import datetime

# load dataset and split
    
def dataset_split(X,Y):  
    train_split = int(len(X)/10 * 9)
    val_split = int(len(X)/10 * 9.5) 
    train_X, train_Y = X[:train_split], Y[:train_split]
    val_X, val_Y = X[train_split:val_split], Y[train_split:val_split]
    test_X, test_Y = X[val_split:], Y[val_split:]
    
    return train_X, train_Y, val_X, val_Y, test_X, test_Y


def read_csv(path):
    rows = []
    with open(path, 'r',  encoding='utf-8') as file:
        csvreader = csv.reader(file)
        header = next(csvreader)
        for row in csvreader:
            rows.append(row)
    return header, rows