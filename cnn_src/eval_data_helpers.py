#! /usr/bin/env python

import logging
import os.path
import sys
import jieba
import re

logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
logging.root.setLevel(level=logging.INFO)
logger = logging.getLogger()

def process_data(line):
    """
    word break and remove word
    Returns split sentences
    """
    # Word break
    seg_list = jieba.cut(line)
    line = u' '.join(seg_list)
    # Remove word
    ss = re.findall('[\n\s*\r\u4e00-\u9fa5]|nmovie|nrcelebrity', line)
    line = u"".join(ss).strip()

    if(len(line) < 2):
        return "UNK"
    return line

def load_data(eval_data_file):
    eval_data = list(open(eval_data_file, "r").readlines())
    row_data = [s.strip().split("\t") for s in eval_data]
    X = [process_data(item[0]) for item in row_data]
    Y = [int(item[1]) for item in row_data]
    return [len(X), X, Y]

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
