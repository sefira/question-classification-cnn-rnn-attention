#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import logging
import data_helpers
from word2vec_helpers import Word2VecHelper
from text_cnn import TextCNN
from tensorflow.contrib import learn
from tensorflow.python.platform import gfile

# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", .1,          "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("train_data_file",       "../data/zhidao_dataneg.tsv.removeword", "Data source for the train data.")
tf.flags.DEFINE_string("raw_data_file",         "../data/zhidao_dataneg.tsv", "Data source for the label data.")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim",        300,        "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes",          "3,4,5",    "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters",          32,         "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob",      0.5,        "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda",          0.0,        "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size",           128,        "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs",           1,          "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every",       100,        "Evaluate model after X steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every",     5000,       "Save model after X steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints",      5,          "Number of checkpoints to store (default: 5)")

# Misc Parameters
tf.flags.DEFINE_string("checkpoint",            '',         "Resume checkpoint")
tf.flags.DEFINE_boolean("allow_soft_placement", True,       "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False,      "Log placement of ops on devices")

logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
logging.root.setLevel(level=logging.INFO)
logger = logging.getLogger()

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
logger.info("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    logger.info("{}={}".format(attr.upper(), value))
logger.info("")


# Data Preparation
# ==================================================

# Load data
logger.info("Loading data...")
train_size, train_data, train_label = data_helpers.load_data(FLAGS.train_data_file, FLAGS.raw_data_file)
x_text = train_data
y = np.array(train_label)

# Build vocabulary
max_document_length = max([len(x.split()) for x in train_data])
word2vec_helpers = Word2VecHelper()
x = word2vec_helpers.SentencesIndex(x_text, max_document_length)
#vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
#x = np.array(list(vocab_processor.fit_transform(x_text)))

# Randomly shuffle data
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]

# Split train/test set
# TODO: This is very crude, should use cross-validation
dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
logger.info("Vocabulary Size: {:d}".format(word2vec_helpers.vocab_size))
logger.info("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))


# Training
# ==================================================

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
    allow_soft_placement=FLAGS.allow_soft_placement,
    log_device_placement=FLAGS.log_device_placement)
    session_conf.gpu_options.allow_growth = True
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = TextCNN(
            sequence_length=x_train.shape[1],
            num_classes=y_train.shape[1],
            vocab_size=word2vec_helpers.vocab_size,
            embedding_size=FLAGS.embedding_dim,
            filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
            num_filters=FLAGS.num_filters,
            l2_reg_lambda=FLAGS.l2_reg_lambda,
            vocab_array=word2vec_helpers.wordvector.astype(np.float32)
            )

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        # Output directory for models and summaries
        if FLAGS.checkpoint == "":
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            logger.info("Writing to {}\n".format(out_dir))
        else:
            out_dir = FLAGS.checkpoint

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", cnn.loss)
        acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Dev summaries
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

        # Write vocabulary
        #vocab_processor.save(os.path.join(out_dir, "vocab"))

        # Initialize all variables
        sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state(os.path.join(FLAGS.checkpoint, 'checkpoints'))
        if ckpt and gfile.Exists(ckpt.model_checkpoint_path):
            print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
            saver.restore(sess, ckpt.model_checkpoint_path)

        def train_step(x_batch, y_batch):
            """
            A single training step
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
            }
            _, step, summaries, loss, accuracy = sess.run(
                [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            logger.info("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step)

        def dev_step(x_batch, y_batch, writer=None):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: 1.0
            }
            step, summaries, loss, accuracy = sess.run(
                [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            logger.info("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            if writer:
                writer.add_summary(summaries, step)

        # Generate batches
        batches = data_helpers.batch_iter(
            list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
        logger.info("With {} batch size, and {} train samples".format(FLAGS.batch_size, len(x_train)))
        logger.info("We get {} batches per epoch".format(len(x_train)/FLAGS.batch_size))

        # Training loop. For each batch...
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:
                logger.info("\nEvaluation:")
                dev_step(x_dev, y_dev, writer=dev_summary_writer)
                logger.info("")
            if current_step % FLAGS.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                logger.info("Saved model checkpoint to {}\n".format(path))

        logger.info("Final Checkpoint:")
        path = saver.save(sess, checkpoint_prefix, global_step=current_step)
        logger.info("Saved model checkpoint to {}\n".format(path))
