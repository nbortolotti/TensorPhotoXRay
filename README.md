# AppEngine Flexible Environtment for TensorFlow Photo Analysis

To use this demonstration, the TensorFlow Object Detection API should be used. For more details here.

To use this example you will need these minimum elements of TensorFlow Object Detection API:

* Models [folder]
* Proto [folder]
* Utils [folder]

Note: if you need download the functional file, [here](https://storage.googleapis.com/appengine_tensorflow/object_detection_api.zip) I prepare a working version.

In addition we need to have the frozen_inference_graph of the model in our solution. For our example:
* Ssd_mobilenet_v1_coco_11_06_2017

Note: if you need download the functional file, [here](https://storage.googleapis.com/appengine_tensorflow/ssd_mobilenet_v1_coco_11_06_2017.zip) I prepare a working version.

We for our example we use COCO as an object recognition scheme, we need the recognition tags for our analysis.
* data/mscoco_label_map.pbtxt

Note: if you need download the functional file, [here](https://storage.googleapis.com/appengine_tensorflow/data.zip) I prepare a working version.

The common structure of the project would be:
![common structure](https://storage.googleapis.com/appengine_tensorflow/post_appengine_tensor_1.png)