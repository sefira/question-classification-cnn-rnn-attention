import eval_data_helpers
from word2vec_helpers import Word2VecHelper

from grpc.beta import implementations
import tensorflow as tf
import numpy as np
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2

from flask import Flask
from flask import request

tf.app.flags.DEFINE_string("server",        "localhost:9001",       "PredictionService host:port")
tf.app.flags.DEFINE_string("eval_data_file","/home/t-xibu/question-classification-cnn-tf/data/eval_data.txt",        "path to image in JPEG format")
tf.app.flags.DEFINE_boolean("eval",         False,                  "Evaluate in eval data")
FLAGS = tf.app.flags.FLAGS

host, port = FLAGS.server.split(":")
channel = implementations.insecure_channel(host, int(port))
stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

max_document_length = 22
word2vec_helpers = Word2VecHelper()
predict_request = predict_pb2.PredictRequest()

def eval():
    # Load data
    eval_size, x_raw, y_test = eval_data_helpers.load_data(FLAGS.eval_data_file)
    x_test = word2vec_helpers.SentencesIndex(x_raw, max_document_length)

    all_predictions = []
    all_scores = []
    all_softmax = []
    # Send predict_request
    for x in x_test:
        input_x = x.tolist()
        # print(input_x)
        # print(type(input_x))
        dropout_keep_prob = 1.0

        predict_request.model_spec.name = "CNN_classifier"
        predict_request.model_spec.signature_name = "predict_sentence"
        # print(tf.contrib.util.make_tensor_proto([input_x], shape=[1,22], dtype=tf.int32))

        predict_request.inputs["input_x"].CopyFrom(
            tf.contrib.util.make_tensor_proto([input_x], shape=[1,22], dtype=tf.int32))
        predict_request.inputs["dropout_keep_prob"].CopyFrom(
            tf.contrib.util.make_tensor_proto(dropout_keep_prob, shape=[1], dtype=tf.float32))
        result = stub.Predict(predict_request, 10.0)  # 10 secs timeout
        # print(type(result))
        feature_configs = {
            "prediction": tf.FixedLenFeature(shape=[], dtype=tf.int64),
            "scores": tf.FixedLenFeature(shape=[], dtype=tf.float32),
            "softmax": tf.FixedLenFeature(shape=[], dtype=tf.float32),
        }
        prediction = result.outputs['prediction'].int64_val
        scores = np.array(result.outputs['scores'].float_val)
        softmax = np.array(result.outputs['softmax'].float_val)
        # print(prediction)
        # print(scores)
        # print(softmax)
        all_predictions = np.concatenate([all_predictions, prediction])
        all_scores.append(scores)
        all_softmax.append(softmax)
        #break

    # Print accuracy if y_test is defined
    if y_test is not None:
        correct_predictions = float(sum(all_predictions == y_test))
        print("Total number of test examples: {}".format(len(y_test)))
        print("Accuracy: {:g}".format(correct_predictions/float(len(y_test))))
        for th in np.linspace(0,0.95,10):
            threshold = th
            true_pos = 0
            true_neg = 0
            false_pos = 0
            false_neg = 0
            for i in range(len(y_test)):
                if all_predictions[i] != 0:
                    if all_softmax[i][int(all_predictions[i])] > threshold:
                        if all_predictions[i] == y_test[i]: 
                            true_pos += 1
                        if all_predictions[i] != y_test[i]:
                            false_pos += 1

            precision = true_pos / (true_pos + false_pos)
            print("Precision: {} in {} threshold".format(precision, threshold))

def predict(line):
    # Load data
    x = eval_data_helpers.process_data(line)
    xs = word2vec_helpers.SentencesIndex([x], max_document_length)

    # Send predict_request
    if len(xs) > 0:
        x = xs[0]
        input_x = x.tolist()
        # print(input_x)
        # print(type(input_x))
        dropout_keep_prob = 1.0

        predict_request.model_spec.name = "CNN_classifier"
        predict_request.model_spec.signature_name = "predict_sentence"

        predict_request.inputs["input_x"].CopyFrom(
            tf.contrib.util.make_tensor_proto([input_x], shape=[1,22], dtype=tf.int32))
        predict_request.inputs["dropout_keep_prob"].CopyFrom(
            tf.contrib.util.make_tensor_proto(dropout_keep_prob, shape=[1], dtype=tf.float32))
        result = stub.Predict(predict_request, 1.0)  # 10 secs timeout
        # print(type(result))
        feature_configs = {
            "prediction": tf.FixedLenFeature(shape=[], dtype=tf.int64),
            "scores": tf.FixedLenFeature(shape=[], dtype=tf.float32),
            "softmax": tf.FixedLenFeature(shape=[], dtype=tf.float32),
        }
        prediction = result.outputs['prediction'].int64_val
        scores = np.array(result.outputs['scores'].float_val)
        softmax = np.array(result.outputs['softmax'].float_val)
        return [prediction, scores, softmax]
    return None


web_server = Flask(__name__)

@web_server.route('/')
def hello_world():
    return 'Hello World!'

@web_server.route("/task/")
def task_list():
    return "List of all task"

@web_server.route("/query")
def get_query():
    query = request.args.get('q')
    result = predict(query)
    threshold = 0.95
    if (len(result) == 3) and \
    (len(result[0]) == 1) and (len(result[1]) == 8) and (len(result[2]) == 8):
        prediction = result[0][0]
        scores = result[1]
        softmax = result[2]
        if softmax[int(prediction)] > threshold:
            print(str(prediction))
            return str(prediction)
        else:
            return str(-1)
    else:
        return str(-1)

    return str(-1)

def main(_):
    if FLAGS.eval:
        eval()
    
    web_server.run(host='0.0.0.0', port=9005, debug=True)

if __name__ == "__main__":
    tf.app.run()
