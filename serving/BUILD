# Description: Tensorflow Serving examples.

package(
    default_visibility = ["//tensorflow_serving:internal"],
    features = ["no_layering_check"],
)

licenses(["notice"])  # Apache 2.0

load("//tensorflow_serving:serving.bzl", "serving_proto_library")

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
)

#### MY CNN QUESTION CLASSIFIER ####
py_library(
    name = "eval_data_helpers",
    srcs_version = "PY2AND3",
    srcs = ["eval_data_helpers.py"],
)
py_library(
    name = "word2vec_helpers",
    srcs_version = "PY2AND3",
    srcs = ["word2vec_helpers.py"],
)
py_binary(
    name = "export_tfserving",
    srcs = [
        "export_tfserving.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)
py_binary(
    name = "export_tfserving_client",
    srcs = [
        "export_tfserving_client.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":eval_data_helpers",
        ":word2vec_helpers",
        "//tensorflow_serving/apis:predict_proto_py_pb2",
        "//tensorflow_serving/apis:prediction_service_proto_py_pb2",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)
####################################

py_library(
    name = "mnist_input_data",
    srcs_version = "PY2AND3",
    srcs = ["mnist_input_data.py"],
)

# TODO(b/32628014): remove mnist_export after we no longer support
# SessionBundle.
py_binary(
    name = "mnist_export",
    srcs = [
        "mnist_export.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":mnist_input_data",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "@org_tensorflow//tensorflow/contrib/session_bundle:exporter",
    ],
)

py_binary(
    name = "mnist_saved_model",
    srcs = [
        "mnist_saved_model.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":mnist_input_data",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_binary(
    name = "mnist_saved_model_py3",
    srcs = [
        "mnist_saved_model_py3.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":mnist_input_data",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_binary(
    name = "mnist_client",
    srcs = [
        "mnist_client.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":mnist_input_data",
        "//tensorflow_serving/apis:predict_proto_py_pb2",
        "//tensorflow_serving/apis:prediction_service_proto_py_pb2",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

# TODO(b/32628014): remove inception_export after we no longer support
# SessionBundle.
py_binary(
    name = "inception_export",
    srcs = [
        "inception_export.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "@inception_model//inception",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "@org_tensorflow//tensorflow/contrib/session_bundle:exporter",
    ],
)

py_binary(
    name = "inception_saved_model",
    srcs = [
        "inception_saved_model.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "@inception_model//inception",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_binary(
    name = "inception_saved_model_py3",
    srcs = [
        "inception_saved_model_py3.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "@inception_model//inception",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_binary(
    name = "inception_client",
    srcs = [
        "inception_client.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow_serving/apis:predict_proto_py_pb2",
        "//tensorflow_serving/apis:prediction_service_proto_py_pb2",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

cc_binary(
    name = "inception_client_cc",
    srcs = [
        "inception_client.cc",
    ],
    deps = [
        "//tensorflow_serving/apis:prediction_service_proto",
        "@grpc//:grpc++",
        "@org_tensorflow//tensorflow/core:framework",
        "@protobuf//:protobuf_lite",
    ],
)
