import numpy as np
import re
import itertools
import logging
# for over-sampling imbalanced learning
from collections import Counter
import numpy as np
from sklearn.utils import check_random_state
from scipy.sparse import hstack,vstack

logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
logging.root.setLevel(level=logging.INFO)
logger = logging.getLogger()

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

class Sample():
    def __init__(self, train_data, row_data, label):
        self.train_data = train_data
        self.row_data = row_data
        self.label = label

def read_data_and_labels(train_data_file, row_data_file):
    """
    Loads data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    train_data = list(open(train_data_file, "r").readlines())
    train_data = [s.strip() for s in train_data]
    row_data = list(open(row_data_file, "r").readlines())
    row_data = [s.strip().split("\t") for s in row_data]
    
    train_number = len(train_data)
    logger.info("Read {} lines data".format(train_number))
    if (len(train_data) != len(row_data)):
        logger.info("Can't read data, trian data is inconsistent with row data")
        return None
    
    train_samples = []
    train_labels = []
    for i in range(len(train_data)):
        if (len(row_data[i]) != 2):
            logger.info("Trian data and label data can't be parsed in {}: {}".format(i,row_data[i]))
            train_number = train_number - 1
            continue
        if (len(train_data[i]) <= 1):
            logger.info("Invalid train data in {}: {}".format(i, train_data[i]))
            train_number = train_number - 1
            continue
        train_data_temp = train_data[i]
        row_data_temp = row_data[i][0]
        label_temp = int(row_data[i][1])
        train_samples.append(Sample(train_data_temp, row_data_temp, label_temp))
        train_labels.append(label_temp)
    return [train_number, train_samples, train_labels]

def oversample_data(X, y):
    if (len(X) != len(y)):
        logger.info("Can't Resample, number is inconsistent:{}:{}".format(len(X), len(y)))
        return None
    y = np.array(y)
    logger.info('Original dataset shape {}'.format(Counter(y)))
    """Resample the dataset.
    """
    label = np.unique(y)
    count_per_label = {}
    majority_count = 0
    for i in label:
        this_count = sum(1 for item in y if item==i)
        count_per_label[i] = this_count
        if this_count > majority_count:
            majority_count = this_count 
            majority_label = i
    # Keep the samples from the majority class
    logger.info("The majority label is {}".format(majority_label))
    X_resampled = [item for item in X if item.label == majority_label]
    y_resampled = [item for item in y if item == majority_label]
 
    # Loop over the other classes over picking at random
    for key in count_per_label.keys():
        # If this is the majority class, skip it
        if key == majority_label:
            continue
 
        # Define the number of sample to create
        num_samples = int(count_per_label[majority_label] - count_per_label[key])
        X_thislabel = [item for item in X if item.label == key]
        # Pick some elements at random
        random_state = check_random_state(42)
        indx = random_state.randint(low=0, high=count_per_label[key],size=num_samples)
        X_sampling = [ X_thislabel[i] for i in indx ]
        # Concatenate to the majority class
        X_resampled = X_resampled + X_thislabel + X_sampling
        logger.info("label: {}, #origin: {}, #sampling: {}".format(key,np.shape(y[y == key]),np.shape(y[y == key][indx])))
        y_resampled = list(y_resampled)+list(y[y == key])+list(y[y == key][indx])
    logger.info('Resampled dataset shape {}'.format(Counter(y_resampled)))
    logger.info("Over-Sampling is complete, total #{} samples".format(len(X_resampled)))
    
    return [X_resampled, y_resampled]

num2onehot={
    -1: [1,0,0,0,0,0,0,0],
    0:  [0,1,0,0,0,0,0,0],
    1:  [0,0,1,0,0,0,0,0],
    2:  [0,0,0,1,0,0,0,0],
    3:  [0,0,0,0,1,0,0,0],
    4:  [0,0,0,0,0,1,0,0],
    6:  [0,0,0,0,0,0,1,0],
    7:  [0,0,0,0,0,0,0,1]
}

label_num = [-1, 0, 1, 2, 3, 4, 6, 7]

def load_data(train_data_file, row_data_file):
    N, X, y = read_data_and_labels(train_data_file, row_data_file)
    if (len(X) != len(y) or len(X) != N or N != len(y)):
        logger.info("Can't load data, number is inconsistent:{}:{}:{}".format(N, len(X), len(y)))
        return None
    
    #oversampling data to against imbalanced data
    X, y = oversample_data(X, y)

    # format data and label 
    """
    transfer label 		from 	to 	formated
    Negative			-1		0	[1,0,0,0,0,0,0,0]
    Movie_Artists		0		1	[0,1,0,0,0,0,0,0]
    Movie_Directors		1		2	[0,0,1,0,0,0,0,0]
    Movie_PublishDate	2		3	[0,0,0,1,0,0,0,0]
    //Movie_Rating	
    Movie_Genres		3		4	[0,0,0,0,1,0,0,0]
    Movie_Country		4		5	[0,0,0,0,0,1,0,0]
    Movie_Description
    Celebrity_Act		6		6	[0,0,0,0,0,0,1,0]
    Celebrity_Direct	7		7	[0,0,0,0,0,0,0,1]
    """

    y_formated = []
    X_formated = []
    for i in range(len(X)):
        if (X[i].label != y[i]):
            logger.info("Data or Label is inconsistent in {}".format(i))
            return None
        if (X[i].label in num2onehot):
            y_formated.append(num2onehot[X[i].label])
            X_formated.append(X[i].train_data)

    if (len(X_formated) != len(y_formated)):
        logger.info("Number is inconsistent after num2onehot")
        return None
    N = len(X_formated)

    # check if there is any wrong in the formated data and label
    random_state = check_random_state(12)
    check_index = random_state.randint(low=0, high=N-1,size=1000)
    for i in check_index:
        if X[i].train_data != X_formated[i]:
            logger.info("Data is inconsistent after num2onehot :{}:{}".format(X[i].train_data, X_formated[i]))
            return None
        if label_num.index(X[i].label) != y_formated[i].index(1):
            logger.info("Label is inconsistent after num2onehot")
            return None

    return [N, X_formated, y_formated]

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
