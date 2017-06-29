# rm -rf /tmp/CNN_classifier_export
# bazel build //tensorflow_serving/example:export_tfserving
# bazel-bin/tensorflow_serving/example/export_tfserving
# bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server --port=9000 --model_name=CNN_classifier --model_base_path=/tmp/CNN_classifier_export

import tensorflow as tf
import numpy as np
import os
import time
import datetime
from tensorflow.contrib import learn
from tensorflow.python.platform import gfile

# Parameters
# ==================================================

tf.app.flags.DEFINE_string("checkpoint_dir",        "/home/t-xibu/question-classification-cnn-tf/src/runs/1497870593",      "Checkpoint directory from training run")
tf.app.flags.DEFINE_string("output_dir",            "/tmp/CNN_classifier_export","Directory where to export inference model.")
tf.app.flags.DEFINE_string("version",               "1",                    "version which is integrant when export folder")
tf.app.flags.DEFINE_boolean("allow_soft_placement", True,                   "Allow device soft device placement")
tf.app.flags.DEFINE_boolean("log_device_placement", False,                  "Log placement of ops on devices")

FLAGS = tf.app.flags.FLAGS

def export():

    print("\nExporting...\n")

    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement)
        session_conf.gpu_options.allow_growth = True
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            # Checkpoint
            ckpt = tf.train.get_checkpoint_state(os.path.join(FLAGS.checkpoint_dir, 'checkpoints'))
            if ckpt:
                print("Read model parameters from %s" % ckpt.model_checkpoint_path)
                # Load the saved meta graph and restore variables
                saver = tf.train.import_meta_graph("{}.meta".format(ckpt.model_checkpoint_path))
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                print('No checkpoint file found at %s' % FLAGS.checkpoint_dir)
                return

            # Tensor we want to input
            # serialized_input_x = tf.placeholder(tf.int64, name='tf_example_input_x')
            # feature_configs_input_x = {'x': tf.FixedLenFeature(shape=[], dtype=tf.int64),}
            # tf_example_input_x = tf.parse_example(serialized_input_x, feature_configs_input_x)
            # input_x = tf.identity(tf_example_input_x['x'], name='input_x')
            # input_x = tf.placeholder(tf.int32, name='input_x')
            # dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')

            # Get the placeholders from the graph by name
            # input_x = tf.placeholder(tf.int32, shape=[None,22], name="input_x")
            input_x = graph.get_operation_by_name("input_x").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

            # Tensors we want to evaluate
            predictions = graph.get_operation_by_name("output/predictions").outputs[0]
            scores = graph.get_operation_by_name("output/scores").outputs[0]
            softmax = tf.nn.softmax(scores)

            # Export inference model.
            output_path = os.path.join(FLAGS.output_dir, FLAGS.version)
            print('Exporting trained model to', output_path)
            builder = tf.saved_model.builder.SavedModelBuilder(output_path)

            input_tensor_info = tf.saved_model.utils.build_tensor_info(input_x)
            dropout_tensor_info = tf.saved_model.utils.build_tensor_info(dropout_keep_prob)
            prediction_tensor_info = tf.saved_model.utils.build_tensor_info(predictions)
            score_tensor_info = tf.saved_model.utils.build_tensor_info(scores)
            softmax_tensor_info = tf.saved_model.utils.build_tensor_info(softmax)

            prediction_signature = (
                tf.saved_model.signature_def_utils.build_signature_def(
                    inputs={'input_x': input_tensor_info,
                            'dropout_keep_prob': dropout_tensor_info
                    },
                    outputs={
                        'prediction': prediction_tensor_info,
                        'scores': score_tensor_info,
                        'softmax': softmax_tensor_info
                    },
                    method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
                ))

            legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
            builder.add_meta_graph_and_variables(
                sess, [tf.saved_model.tag_constants.SERVING],
                signature_def_map={
                    'predict_sentence':
                        prediction_signature,
                    # tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                    #     classification_signature,
                },
                legacy_init_op=legacy_init_op)
            builder.save()
            print('Successfully exported model to %s' % FLAGS.output_dir)

def main(unused_argv=None):
  export()

if __name__ == '__main__':
  tf.app.run()
