import logging
import os
import sys
import multiprocessing
import numpy as np
from gensim.models import Word2Vec
from sklearn.utils import check_random_state

#################### config ###################
modelfile = "../wvmodel/size300window5sg1min_count100negative10iter50.model"
questionfile = "../wvmodel/questions-words-Zh.txt"
############### end of config #################

logger = logging.getLogger()
logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
logging.root.setLevel(level=logging.INFO)
logger.info("running test word2vec for model: %s" % modelfile)


class Word2VecHelper(object):
    """
    import word2vec from gensim
    """
    def __init__(self, test_model=False, verify_model=True):
        model = Word2Vec.load(modelfile)

        if(test_model):
            acc = model.accuracy(questionfile)
            logger.info("Test model " + modelfile + " in " + questionfile)

        self.vector_size = model.vector_size
        self.vocab_size = len(model.wv.vocab) + 1
        self.word2index = self.GetWord2Index(model)
        self.index2word = self.GetIndex2Word(model)
        self.wordvector = self.GetWordVector(model)

        if(verify_model):
            logger.info("Verifing imported word2vec model")
            random_state = check_random_state(12)
            check_index = random_state.randint(low=0, high=self.vocab_size-2,size=1000)
            for index in check_index:
                word_wv = model.wv.index2word[index]
                word_our = self.index2word[index+1]
                #print(index, word_wv, word_our)
                assert word_wv == word_our
                assert model.wv.vocab[word_our].index == self.word2index[word_our] - 1
                assert np.array_equal(model.wv[word_our], self.wordvector[self.word2index[word_our]])
            logger.info("Imported word2vec model is verified")

    def GetWord2Index(self, model):
        word2index = {}
        word2index["UNK"] = 0
        for key, value in model.wv.vocab.items():
            word2index[key] = value.index + 1

        if(len(word2index) != self.vocab_size):
            logger.error("Get word2index error")
            return None
        logger.info("Got Word2Index")
        return word2index

    def GetIndex2Word(self, model):
        index2word = ["UNK"] + model.wv.index2word

        if(len(index2word) != self.vocab_size):
            logger.error("Get index2word error")
            return None
        logger.info("Got Index2Word")
        return index2word

    def GetWordVector(self, model):
        wordvector = np.zeros((self.vocab_size, self.vector_size))
        wordvector[0] = np.zeros((self.vector_size))
        count = 0
        for word in model.wv.index2word:
            wordvector[count + 1] = model.wv[word]
            count = count + 1

        if(len(wordvector) != self.vocab_size):
            logger.error("Get WordVector error")
            return None
        logger.info("Got WordVector")
        return wordvector

    def SentencesIndex(self, sentences, max_document_length):
        # indexed_sentences = np.zeros((len(sentences), max_document_length), np.int64)
        # for count, sentence in enumerate(sentences):
        #     sentence_split = sentence.split()
        #     for index, word in enumerate(sentence_split):
        #         if word in self.word2index:
        #             indexed_sentences[count][index] = (self.word2index[word])
        #         else:
        #             indexed_sentences[count][index] = 0
        indexed_sentences = np.array(
            [[self.word2index[word] for word in sentence.split() if word in self.word2index ] for sentence in sentences]
            )
        logger.info("{} Sentences have been indexed".format(len(indexed_sentences)))
        indexed_sentences = np.array([x[:max_document_length - 1] + [0] * max(max_document_length - len(x), 1) for x in indexed_sentences])
        return indexed_sentences
                    
if __name__ == '__main__':
    model = Word2Vec.load(modelfile)
    word2vec_helpers = Word2VecHelper()
