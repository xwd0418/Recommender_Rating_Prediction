import random

# load dataset
def load_dataset():
    """loading dataset

    Returns:
        all_data, all_labels
    """

    all_data = []
    # TODO: downloading dataset and put data into a list
    # each item in the list all_data should be a tuple with 2 elements
    # the first element is a dict of data details (recipe id, name, steps, etc)
    # the second element is the lable (rating)
    random.shuffle(all_data)
    X, Y = zip(*all_data) 
    return X, Y

def dataset_split():
    X, Y = load_dataset()
    
    train_split = len(X)//10 * 8
    val_split = len(X)//10 * 9
    
    train_X, train_Y = X[:train_split], Y[:train_split]
    val_X, val_Y = X[train_split:val_split], Y[train_split:val_split]
    test_X, test_Y = X[val_split:], Y[:val_split]
    
    return train_X, train_Y, val_X, val_Y, test_X, test_Y

