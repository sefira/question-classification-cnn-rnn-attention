## CNN, RNN and Attention model for Chinese Question Classification in Tensorflow
## 中文语料(百度知道电影领域)问题分类模型 CNN, RNN and Attention modelin TensorFlow
***
This project forked from [dennybritz/cnn-text-classification-tf](https://github.com/dennybritz/cnn-text-classification-tf), [jiegzhan/multi-class-text-classification-cnn-rnn](https://github.com/jiegzhan/multi-class-text-classification-cnn-rnn) and [ilivans/tf-rnn-attention](https://github.com/ilivans/tf-rnn-attention) makes the data helper supports Chinese language and modified the embedding from random to gensim pre-trained. This version can achieve an accuracy of 82% with the Chinese zhidao.baidu.com movie corpus

This algorithm refers to the 
- [Implementing a CNN for Text Classification in Tensorflow blog post.](http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/)
- [Convolutional Neural Networks for Sentence Classification](http://arxiv.org/abs/1408.5882)
- [A Sensitivity Analysis of (and Practitioners' Guide to) Convolutional Neural Networks for Sentence Classification](http://arxiv.org/abs/1510.03820)

## Requirements

- Python 3.6
- Anaconda 4.4.0
- Tensorflow 1.1.0

## Preparation
- Processs Chinese Corpus in the same manner with my [word2vec project](https://github.com/sefira/word2vec)
```bash
python -i process_corpus.py
```

- Place pre-trained Word2Vec model in the wvmodel folder, so that code can read it before train CNN.

## Training
```bash
python -i train.py
```
While running train.py, it will:
- import pre-trained word2vec model and use the vocabulary to initialize the CNN input in word2vec_helpers.py
- create a CNN and RNN model which is defined in text_cnn.py
- read the trian data which has been read from file and over-sampled and shuffled in data_helpers.py
- train a CNN and RNN model using above mentioned material

Training arrives its end after one epoch, in other word, 16874 batches with batch size 128 and train data 2159871 samples

## Visualization
After training, we can plot some intuitive pictures of training summary in TensorBoard
```bash
tensorboard --logdir src/runs
```
Accuracy 82% in CNN
![Accuracy 82%](https://raw.githubusercontent.com/sefira/question-classification-cnn-tf/master/SCREENSHOT/SCALARScnn.png)
CNN Model Graph
![CNN Model Graph](https://raw.githubusercontent.com/sefira/question-classification-cnn-tf/master/SCREENSHOT/GRAPHScnn.png)
Model Parameter Histogram
![Parameter Histogram](https://raw.githubusercontent.com/sefira/question-classification-cnn-tf/master/SCREENSHOT/HISTOGRAMScnn.png)

Accuracy 82% in RNN with Attention
![Accuracy 82%](https://raw.githubusercontent.com/sefira/question-classification-cnn-tf/master/SCREENSHOT/SCALARSrnn.png)
CNN Model Graph
![CNN Model Graph](https://raw.githubusercontent.com/sefira/question-classification-cnn-tf/master/SCREENSHOT/GRAPHSrnn.png)
Model Parameter Histogram

## Evaluating
```bash
python -i eval.py
```
After all training process, we can evaluate the trained model in natural language test data. The evaluate batch is 1, so that this code can be easily transferred to TensorFlow Serving saved_model.

***
本代码Fork自[dennybritz/cnn-text-classification-tf](https://github.com/dennybritz/cnn-text-classification-tf), [jiegzhan/multi-class-text-classification-cnn-rnn](https://github.com/jiegzhan/multi-class-text-classification-cnn-rnn) and [ilivans/tf-rnn-attention](https://github.com/ilivans/tf-rnn-attention)。不过与其不同的是我使用data_helprs.py中支持了中文处理，并且修改了词嵌入层，将原来的随机化词嵌入改为了使用预训练好的word2vec向量。也对模型代码和训练过程参数有一些调整。这个版本的问题分类模型在百度知道电影问题分类上实现了82%的准确度

本算法参考了 
- [Implementing a CNN for Text Classification in Tensorflow blog post.](http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/)
- [Convolutional Neural Networks for Sentence Classification](http://arxiv.org/abs/1408.5882)
- [A Sensitivity Analysis of (and Practitioners' Guide to) Convolutional Neural Networks for Sentence Classification](http://arxiv.org/abs/1510.03820)

## 安装

- Python 3.6
- Anaconda 4.4.0
- Tensorflow 1.1.0

## 预备工作
- 使用和我之前[word2vec project](https://github.com/sefira/word2vec)中类似的方法处理训练语料
```bash
python -i process_corpus.py
```

- 将训练好的word2vec模型放置在wvmodel文件夹中，这样在初始化CNN的时候词嵌入层才能从word2vec中导入词向量，而不是随机初始化

## 训练
```bash
python -i train.py
```
当我们运行train.py时，它会执行以下操作:
- 导入word2vec_helpers.py读取好的预训练word2vec模型，该模型之后被用作初始化CNN输入词向量
- 创建text_cnn.py中定义好的CNN分类模型
- 读取data_helpers.py中预处理并且重采样好的训练数据，以batch的方式迭代读入，并对数据随机打散优化训练过程
- 使用上述材料训练CNN问题分类器

训练经历一个世代后模型收敛。因为采用了128的batch size和2159871个样本，所以一个世代经历了16874个batch

## 可视化结果
训练之后我们在TensorBoard上可视化训练结果
```bash
tensorboard --logdir src/runs
```
可视化后可以看到实现了82%准确率，CNN模型图，模型参数直方图
Accuracy 82%
![Accuracy 82%](https://raw.githubusercontent.com/sefira/question-classification-cnn-tf/master/SCREENSHOT/SCALARS.png)
CNN Model Graph
![CNN Model Graph](https://raw.githubusercontent.com/sefira/question-classification-cnn-tf/master/SCREENSHOT/GRAPHS.png)
Model Parameter Histogram
![Parameter Histogram](https://raw.githubusercontent.com/sefira/question-classification-cnn-tf/master/SCREENSHOT/HISTOGRAMS.png)

使用RNN和Attention实现了82%的准确率
![Accuracy 82%](https://raw.githubusercontent.com/sefira/question-classification-cnn-tf/master/SCREENSHOT/SCALARSrnn.png)
CNN Model Graph
![CNN Model Graph](https://raw.githubusercontent.com/sefira/question-classification-cnn-tf/master/SCREENSHOT/GRAPHSrnn.png)
Model Parameter Histogram

## 评估
```bash
python -i eval.py
```
在所有的训练和调参结束后，我们可以在更为困难（更接近真实场景）的自然语言数据上测试模型。
在评估的时候，batch size是1，这样方便将这份代码直接迁移到TensorFlow Serving的saved model中。