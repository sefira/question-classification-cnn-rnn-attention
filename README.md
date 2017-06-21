## CNN for Chinese Question Classification in Tensorflow
Sentiment classification forked from [dennybritz/cnn-text-classification-tf](https://github.com/dennybritz/cnn-text-classification-tf), makes the data helper supports Chinese language and modified the embedding from random to gensim pre-trained. This version can achieve an accuracy of 82% with the Chinese zhidao.baidu.com movie corpus

This code refers to the 
- [Implementing a CNN for Text Classification in Tensorflow blog post.](http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/)
- [Convolutional Neural Networks for Sentence Classification](http://arxiv.org/abs/1408.5882)
- [A Sensitivity Analysis of (and Practitioners' Guide to) Convolutional Neural Networks for Sentence Classification](http://arxiv.org/abs/1510.03820)

It is slightly simplified implementation of Kim's [Convolutional Neural Networks for Sentence Classification](http://arxiv.org/abs/1408.5882) paper in Tensorflow.

## Requirements

- Python 3.6
- Anaconda 4.4.0
- Tensorflow 1.1.0

## Preparation
- Processs Chinese Corpus in the same manner of my word2vec project
```bash
python -i process_corpus.py
```

- Place pre-trained Word2Vec model in the wvmodel folder, so that code can read it before train CNN.

## Training
```bash
python -i train.py
```
While running train.py, it will:
- create a CNN model which is defined in text_cnn.py
- read the trian data which has been read from file and over-sampled in data_helpers.py
- import pre-trained word2vec model and use the vocabulary to initialize the CNN input in word2vec_helpers.py
- train a CNN model use above mentioned material

Training arrives its end after one epoch, in other word, 16874 batches with batch size 128 and train data 2159871 samples

## Visualization
After training, we can plot some intuitive pictures of training summary in TensorBoard
```bash
tensorboard --logdir src/runs
```
![Accuracy 82%](https://raw.githubusercontent.com/sefira/question-classification-cnn-tf/master/SCREENSHOT/SCALARS.py)
![CNN Model Graph](https://raw.githubusercontent.com/sefira/question-classification-cnn-tf/master/SCREENSHOT/GRAPHS.py)
![Parameter Histogram](https://raw.githubusercontent.com/sefira/question-classification-cnn-tf/master/SCREENSHOT/HISTOGRAMS.py)
