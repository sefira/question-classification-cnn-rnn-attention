import tensorflow as tf

class TextRNN(object):
    """
    A RNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(
      self, embedding_mat, non_static, GRU, sequence_length, num_classes, 
      hidden_layer_size, vocab_size,
      embedding_size,attention_size , l2_reg_lambda=0.0):

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.batch_size = tf.placeholder(tf.int32, [], name="batch_size")
        self.real_len = tf.placeholder(tf.int32,[None], name="real_len")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            if non_static:
                W = tf.Variable(embedding_mat, name="W")
            else:
                W = tf.constant(embedding_mat, name="W")
            self.embedded_chars = tf.nn.embedding_lookup(W, self.input_x)

        # Create a rnn layer
        with tf.name_scope("rnn"):
            if GRU:
                rnn_cell = tf.contrib.rnn.GRUCell(num_units=hidden_layer_size)
            else:
                rnn_cell = tf.contrib.rnn.LSTMCell(num_units=hidden_layer_size)
            # rnn_cell = tf.contrib.rnn.DropoutWrapper(
            #     rnn_cell, output_keep_prob=self.dropout_keep_prob)

            self._initial_state = rnn_cell.zero_state(self.batch_size, tf.float32)
            rnn_outputs, rnn_state = tf.nn.dynamic_rnn(
                rnn_cell,
                inputs=self.embedded_chars,
                sequence_length=self.real_len,
                initial_state=self._initial_state
                )

        # An attention model
        with tf.name_scope("attention"):
            # Attention mechanism
            W = tf.Variable(
                tf.truncated_normal([hidden_layer_size, attention_size], 
                stddev=0.1), name="W"
                )
            b = tf.Variable(tf.random_normal([attention_size], stddev=0.1), 
                name="b")
            u = tf.Variable(tf.random_normal([attention_size], stddev=0.1), 
                name="u")

            att = tf.tanh(
                tf.nn.xw_plus_b(tf.reshape(rnn_outputs, [-1, hidden_layer_size]),
                W, b), 
                name="attention_projection"
                )
            logits = tf.matmul(att, tf.reshape(u, [-1, 1]), name="attention_logits")
            attention_weights = tf.nn.softmax(
                tf.reshape(logits, [-1, sequence_length]), 
                dim=1,
                name="attention_weights")

            weighted_rnn_output = tf.multiply(
                rnn_outputs, tf.reshape(attention_weights, [-1, sequence_length, 1]),
                name="weighted_rnn_outputs")
            attention_outputs = tf.reduce_sum(
                weighted_rnn_output, 1, name="attention_outputs")

            dropout_outputs = tf.nn.dropout(
                attention_outputs, self.dropout_keep_prob, 
                name="dropout")

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.Variable(
                tf.truncated_normal(
                    [dropout_outputs.shape[1].value, num_classes], stddev=0.1
                    ), 
                name="W"
                )
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(dropout_outputs, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(
                logits=self.scores, labels=self.input_y
                )
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(
                tf.cast(correct_predictions, "float"), 
                name="accuracy")

