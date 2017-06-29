import eval_data_helpers
from word2vec_helpers import Word2VecHelper

from grpc.beta import implementations
import tensorflow as tf

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2

tf.app.flags.DEFINE_string("server",        "localhost:9000",      "PredictionService host:port")
tf.app.flags.DEFINE_string("eval_data_file","/home/t-xibu/question-classification-cnn-tf/data/eval_data.txt",        "path to image in JPEG format")
FLAGS = tf.app.flags.FLAGS

host, port = FLAGS.server.split(":")
channel = implementations.insecure_channel(host, int(port))
stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

def eval():
    # Load data
    eval_size, x_raw, y_test = eval_data_helpers.load_data(FLAGS.eval_data_file)
    max_document_length = 22
    word2vec_helpers = Word2VecHelper()
    x_test = word2vec_helpers.SentencesIndex(x_raw, max_document_length)

    # Send request
    for x in x_test:
        input_x = x.tolist()
        # print(input_x)
        # print(type(input_x))
        dropout_keep_prob = 1.0

        request = predict_pb2.PredictRequest()
        request.model_spec.name = "CNN_classifier"
        request.model_spec.signature_name = "predict_sentence"
        # print(tf.contrib.util.make_tensor_proto([input_x], shape=[1,22], dtype=tf.int32))

        request.inputs["input_x"].CopyFrom(
            tf.contrib.util.make_tensor_proto([input_x], shape=[1,22], dtype=tf.int32))
        request.inputs["dropout_keep_prob"].CopyFrom(
            tf.contrib.util.make_tensor_proto(dropout_keep_prob, shape=[1], dtype=tf.float32))

        result = stub.Predict(request, 10.0)  # 10 secs timeout
        print(result)

def main(_):
    eval()


if __name__ == "__main__":
    tf.app.run()
