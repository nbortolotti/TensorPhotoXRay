import json
import logging
import cStringIO
import urllib
import os
import sys
import numpy as np
import tensorflow as tf
from PIL import Image


from flask import Flask, send_file, render_template
from utils import label_map_util

app = Flask(__name__)
sys.path.append("..")


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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


@app.route('/photobot/<path:photo_url>')
def tensor_photobot(photo_url):
    file = cStringIO.StringIO(urllib.urlopen(photo_url).read())
    img = Image.open(file)

    if img:
        list_elements = process_image(img)
        return json.dumps(list_elements)


@app.errorhandler(500)
def server_error(e):
    logging.exception('An error occurred during a request.')
    return """
    An internal error occurred: <pre>{}</pre>
    See logs for full stacktrace.
    """.format(e), 500


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080, debug=True)
