# AppEngine Flexible Environment for TensorFlow Photo Analysis

To use this demonstration, the TensorFlow Object Detection API should be used. For more details [here](https://github.com/tensorflow/models/tree/master/object_detection).

To use this example you will need these minimum elements of [TensorFlow Object Detection API](https://github.com/tensorflow/models/tree/master/object_detection):

* Models [folder]
* Proto [folder]
* Utils [folder]

Note: if you need download the functional file, [here](https://storage.googleapis.com/appengine_tensorflow/object_detection_api.zip) I prepare a working version.

In addition we need to have the frozen_inference_graph of the model in our solution. For our example:
* Ssd_mobilenet_v1_coco_11_06_2017

Note: if you need download the functional file, [here](https://storage.googleapis.com/appengine_tensorflow/ssd_mobilenet_v1_coco_11_06_2017.zip) I prepare a working version.

For our example we use COCO as an object recognition scheme, we need the recognition tags for our analysis.
* data/mscoco_label_map.pbtxt

Note: if you need download the functional file, [here](https://storage.googleapis.com/appengine_tensorflow/data.zip) I prepare a working version.

The common structure of the project would be:
![common structure](https://storage.googleapis.com/appengine_tensorflow/post_appengine_tensor_1.png)

# Deploy the project

Note: You should have the [Google Cloud SDK](https://cloud.google.com/sdk/docs/). More information about App Engine Flexible environment, Python [here](https://cloud.google.com/appengine/docs/flexible/python/quickstart)

Local:
python main.py

Production Environment:
gcloud app deploy 

*(-v version) if you want to deploy it to a specific version.