import logging
import os
import sys
import multiprocessing
import numpy as np
from gensim.models import Word2Vec

#################### config ###################
modelfile = "../wvmodel/size300window5sg1min_count100negative10iter50.model"
questionfile = "../wvmodel/questions-words-Zh.txt"
############### end of config #################

logger = logging.getLogger()
logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
logging.root.setLevel(level=logging.INFO)
logger.info("running test word2vec for model: %s" % modelfile)


class MyWord2Vec(object):
    """
    import word2vec from gensim
    """
    def __init__(test_model=True):
        model = Word2Vec.load(modelfile)

        if(test_model):
            acc = model.accuracy(questionfile)
            logger.info("Test model " + modelfile + " in " + questionfile)

        self.vector_size = model.vector_size
        self.vocab_size = len(model.wv.vocab)
        self.word2index = GetWord2Index(model)
        self.index2word = GetIndex2Word(model)
        self.wordvector = GetWordVector(model)

    def GetWord2Index(model):
        word2index = {}
        for key, value in model.wv.vocab.items():
            word2index[key] = value.index

        if(len(word2index) != len(model.wv.vocab)):
            logger.info("Get word2index error")
            return None

        return word2index

    def GetIndex2Word(model):
        index2word = {}
        for key, value in model.wv.index2word.item():
            index2word[key] = value

        if(len(word2index) != len(model.wv.index2word)):
            logger.info("Get index2word error")
            return None

        return index2word

    def GetWordVector(model):
        wordvector = np.array((self.vocab_size, self.vector_size))
        count = 0
        for key, value in model.wv.index2word.item():
            wordvector[key] = model.wv[value]
            count = count + 1

        if(count != len(model.wv.index2word) or (len(wordvector) != len(model.wv.index2word))):
            logger.info("Get WordVector error")
            return None

        return wordvector
