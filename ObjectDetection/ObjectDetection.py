import six.moves.urllib as urllib
import tarfile
import os
import tensorflow as tf
from tensorflow.python.platform import gfile
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import visualization_utils as viz_util
import label_map_util as label_map_util
import cv2
import sys
# http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html

# number_of_arg = len(sys.argv)
# threshold = int(sys.argv[1])
#
# if threshold == 0:
#     threshold = 0.5

MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
PROTOBUF_FILE = 'frozen_inference_graph.pb'
PATH_TO_CKPT = MODEL_NAME + '/' + PROTOBUF_FILE

MODEL_DATA = 'models.ckpt.data-00000-of-00001'
MODEL_INDEX = 'models.ckpt.index'
MODEL_META = 'models.ckpt.meta'
MODEL_LABEL = 'graph.pbtxt'
MODEL_PROTOBUF = 'frozen_inference_graph.pb'

PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
NUM_CLASSES = 90
IMAGE_SIZE = (12, 8)
PATH_TO_TEST_IMAGES_DIR = 'test_images'


def download_frozen_graph(download_url, model_file, protobuf_file):
    opener = urllib.request.URLopener()
    opener.retrieve(download_url + model_file, model_file)
    tar_file = tarfile.open(model_file)
    for file in tar_file.getmembers():
        file_name = os.path.basename(file.name)
        if protobuf_file in file_name:
            tar_file.extract(file, os.getcwd())


def download_frozen_graph_local(model_file):
    tar_file = tarfile.open(model_file)
    for file in tar_file.getmembers():
        file_name = os.path.basename(file.name)
        tar_file.extract(file, os.getcwd())


# References
# https://gist.github.com/tokestermw/795cc1fd6d0c9069b20204cbd133e36b
# https://www.tensorflow.org/api_guides/python/meta_graph
# https://www.tensorflow.org/versions/r0.12/how_tos/tool_developers/
# https://blog.metaflow.fr/tensorflow-how-to-freeze-a-model-and-serve-it-with-a-python-api-d4f3596b3adc
# https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
# https://github.com/tensorflow/models

# All of TensorFlow's file formats are based on Protocol Buffers
# A. GraphDef
# The foundation of computation in TensorFlow is the Graph object.
# This holds a network of nodes, each representing one operation, connected to each other as inputs and outputs.
# After you've created a Graph object, you can save it out by calling as_graph_def(), which returns a GraphDef object
# The GraphDef class is an object created by the ProtoBuf library


def load_graph(frozen_graph_filename):
    with gfile.FastGFile(frozen_graph_filename, "rb") as f:
        # with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        byte = f.read()
        graph_def.ParseFromString(byte)

    # with tf.Graph().as_default() as graph:
    #   tf.import_graph_def(
    #     graph_def,
    #     input_map=None,
    #     return_elements=None,
    #     name="prefix",
    #     op_dict=None,
    #     producer_op_list=None
    #   )

    return graph_def


def load_model(model_meta):
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(model_meta)
        saver.restore(sess, tf.train.latest_checkpoint('./'))
        graph = tf.get_default_graph()

    return graph


def test_image():
    list = os.listdir('test_images')
    PATH_TO_TEST_IMAGES_DIR = 'test_images'
    # print(len(list))
    TEST_IMAGE_PATHS = [os.path.join(PATH_TO_TEST_IMAGES_DIR, list[i]) for i in range(len(list))]

    return TEST_IMAGE_PATHS


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


def load_image_into_numpy_array_frame(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    (im_width, im_height) = np.shape(image)
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)


def run_detection(test_image_path,threshold):
    with tf.Session() as sess:
        detection_graph = tf.get_default_graph()
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
        # weights = detection_graph.get_tensor_by_name('FeatureExtractor/MobilenetV1/Conv2d_0/weights:0')
        # print("images",test_image_path)
        for image_path in test_image_path:
            # print("images")
            image = Image.open(image_path)
            image_np = load_image_into_numpy_array(image)
            # print(np.shape(image_np))
            image_np_expanded = np.expand_dims(image_np, axis=0)
            # print(np.shape(image_np_expanded))
            (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})

            print("======================")
            print("Number of Detection ", num)
            print("Classes  : ", classes)
            print("Scores : ", scores)

            viz_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=8,
                min_score_thresh=threshold)
            plt.figure(figsize=IMAGE_SIZE)
            plt.imshow(image_np)
            plt.show()


def run_detection_camera():
    cam = cv2.VideoCapture(0)

    with tf.Session() as sess:
        detection_graph = tf.get_default_graph()
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
        # weights = detection_graph.get_tensor_by_name('FeatureExtractor/MobilenetV1/Conv2d_0/weights:0')

        while (True):
            ret, frame = cam.read()
            # image_np = load_image_into_numpy_array_frame(frame)
            # print(np.shape(image_np))
            image_np_expanded = np.expand_dims(frame, axis=0)
            (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})

            viz_util.visualize_boxes_and_labels_on_image_array(
                frame,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=8)

            cv2.imshow('Object Detection', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cam.release()
    cv2.destroyAllWindows()


# This line creates an empty GraphDef object, the class that's been created from the textual
# definition in graph.proto.
# This is the object we're going to populate with the data from our file
# detection_graph = tf.Graph()
#
# graph = tf.get_default_graph()
# input_graph_def = graph.as_graph_def()
# with detection_graph.as_default():
#   od_graph_def = tf.GraphDef()
#   with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
#     serialized_graph = fid.read()
#     od_graph_def.ParseFromString(serialized_graph)
#     tf.import_graph_def(od_graph_def, name='')

# graph is loaded in the following variable
# Once you've loaded a file into the graph_def variable, you can now access the data inside i
# building blocks of TensorFlow graphs
# Each node is a NodeDef object, defined in tensorflow/core/framework/node_def.proto.


# Start

# 1. Download Protobuf models graph
# download_frozen_graph(DOWNLOAD_BASE,MODEL_FILE,PROTOBUF_FILE)
# untar_pre_trained_model(MODEL_FILE)

# 2. Load Graph
detection_graph = load_graph('./' + MODEL_NAME + "/" + PROTOBUF_FILE)
# graph_def = load_model("./"+MODEL_NAME+"/"+MODEL_META)

# 3. Import Graph defination
tf.import_graph_def(detection_graph, name='')

# protoc ./*.proto --python_out=.
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# 4. Print nodes present in detection_graph
# for node in detection_graph.node:
#     print(node.name)
# if node.name == 'thresh':
#     print(node.name)

# detection_boxes
# detection_scores
# detection_classes
# num_detections
# image_tensor

# 5. Print operation present in graph
# for op in graph_def.get_operations():
#   print(op.name)
# 6. Load Test Images
# print('Running Object Detection Program')
run_detection_camera()
