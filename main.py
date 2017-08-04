import logging

import cStringIO
import urllib

import os
import sys

import numpy as np
import tensorflow as tf
from PIL import Image
from gcloud import storage

from flask import Flask, send_file, render_template
from flask import jsonify

app = Flask(__name__)


@app.route('/<number>')
def tensor_regression(number):
    train_X = np.array([
        [2.0658746e+00],
        [2.3684087e+00],
        [2.5399929e+00],
        [2.5420804e+00],
        [2.5490790e+00],
        [2.7866882e+00],
        [2.9116825e+00],
        [3.0356270e+00],
        [3.1146696e+00],
        [3.1582389e+00],
        [3.3275944e+00],
        [3.3793165e+00],
        [3.4122006e+00],
        [3.4215823e+00],
        [3.5315732e+00],
        [3.6393002e+00],
        [3.6732537e+00],
        [3.9256462e+00],
        [4.0498646e+00],
        [4.2483348e+00],
        [4.3440052e+00],
        [4.3826531e+00],
        [4.4230602e+00],
        [4.6102443e+00],
        [4.6881183e+00],
        [4.9777333e+00],
        [5.0359967e+00],
        [5.0684536e+00],
        [5.4161491e+00],
        [5.4395623e+00],
        [5.4563207e+00],
        [5.5698458e+00],
        [5.6015729e+00],
        [5.6877617e+00],
        [5.7215602e+00],
        [5.8538914e+00],
        [6.1978026e+00],
        [6.3510941e+00],
        [6.4797033e+00],
        [6.7383791e+00],
        [6.8637686e+00],
        [7.0223387e+00],
        [7.0782373e+00],
        [7.1514232e+00],
        [7.4664023e+00],
        [7.5973874e+00],
        [7.7440717e+00],
        [7.7729662e+00],
        [7.8264514e+00],
        [7.9306356e+00]
    ]
    ).astype('float32')

    mean = np.mean(train_X, axis=0)
    std = np.std(train_X, axis=0)

    # a = tf.placeholder("float")
    # b = tf.placeholder("float")
    #
    # y = tf.multiply(a, b)
    #
    # with tf.Session() as sess:
    #
    #     result = str(sess.run(y, feed_dict={a: 5, b: 5}))
    #
    #     client = storage.Client('lookthiscode-521')
    #     bucket = client.bucket('lookthiscode-521')
    #     blob = bucket.blob('my-model')
    #
    #
    #     #return result
    # inputs

    # X = tf.placeholder(tf.float32, [m, n])
    # Y = tf.placeholder(tf.float32, [m, 1])

    # weight and bias
    # W = tf.Variable(tf.zeros([n, 1], dtype=np.float32), name="weight")
    # b = tf.Variable(tf.zeros([1], dtype=np.float32), name="bias")

    # linear model
    # with tf.name_scope("linear_Wx_b") as scope:
    #    activation = tf.add(tf.matmul(X, W), b)

    # cost
    # with tf.name_scope("cost") as scope:
    #    cost = tf.reduce_sum(tf.square(activation - Y)) / (2 * m)
    #    tf.summary.scalar("cost", cost)

    # train
    # with tf.name_scope("train") as scope:
    #    optimizer = tf.train.GradientDescentOptimizer(0.07).minimize(cost)

    # saver = tf.train.Saver()

    # tensorflow session
    with tf.Session() as sess:

        client = storage.Client('lookthiscode-521')
        bucket = client.bucket('models_lookthiscode')

        meta_file = 'my-model.ckpt.meta'
        if not os.path.isfile(meta_file):
            blob = bucket.blob(meta_file)
            with open(meta_file, 'w') as f:
                blob.download_to_file(f)

        check_file = 'checkpoint'
        if not os.path.isfile(check_file):
            blob1 = bucket.blob(check_file)
            with open(check_file, 'w') as f:
                blob1.download_to_file(f)

        index_file = 'my-model.ckpt.index'
        if not os.path.isfile(index_file):
            blob2 = bucket.blob(index_file)
            with open(index_file, 'w') as f:
                blob2.download_to_file(f)

        data_file = 'my-model.ckpt.data-00000-of-00001'
        if not os.path.isfile(data_file):
            blob3 = bucket.blob(data_file)
            with open(data_file, 'w') as f:
                blob3.download_to_file(f)

        saver = tf.train.import_meta_graph('my-model.ckpt.meta')
        saver.restore(sess, tf.train.latest_checkpoint('./'))

        predict_X_2 = np.array([float(number)], dtype=np.float32).reshape([1, 1])
        predict_X_2 = (predict_X_2 - mean) / std

        graph = tf.get_default_graph()
        W = graph.get_tensor_by_name("weight:0")
        b = graph.get_tensor_by_name("bias:0")

        predict_Y_2 = tf.add(tf.matmul(predict_X_2, W), b)
        return str(sess.run(predict_Y_2))


sys.path.append("..")

from utils import label_map_util

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# MODEL_NAME = 'faster_rcnn_inception_resnet_v2_atrous_coco_11_06_2017'
MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'

MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
NUM_CLASSES = 90

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)


def detect_alert(boxes, classes, scores, category_index, max_boxes_to_draw=20,
                 min_score_thresh=.5,
                 ):
    r = []
    for i in range(min(max_boxes_to_draw, boxes.shape[0])):
        if scores is None or scores[i] > min_score_thresh:
            test1 = None
            test2 = None

            if category_index[classes[i]]['name']:
                test1 = category_index[classes[i]]['name']
                test2 = int(100 * scores[i])

            line = {}
            line[test1] = test2
            r.append(line)

    return r


def detect_objects(image_np, sess, detection_graph):
    image_np_expanded = np.expand_dims(image_np, axis=0)
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    scores = detection_graph.get_tensor_by_name('detection_scores:0')
    classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    # Actual detection.
    (boxes, scores, classes, num_detections) = sess.run(
        [boxes, scores, classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})

    alert_array = detect_alert(np.squeeze(boxes), np.squeeze(classes).astype(np.int32), np.squeeze(scores),
                               category_index)

    return alert_array


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')


def process_image(image):
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            alert_array = detect_objects(image, sess, detection_graph)
            return alert_array


@app.route('/photo/<path:photo_url>')
def tensor_photo(photo_url):
    url = photo_url
    file = cStringIO.StringIO(urllib.urlopen(photo_url).read())
    img = Image.open(file)

    if img:
        list_elements = process_image(img)
        list = str(list_elements)
        return render_template('index.html', **locals())



@app.errorhandler(500)
def server_error(e):
    logging.exception('An error occurred during a request.')
    return """
    An internal error occurred: <pre>{}</pre>
    See logs for full stacktrace.
    """.format(e), 500


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080, debug=True)
