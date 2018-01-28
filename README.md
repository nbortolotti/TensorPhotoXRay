# AppEngine Flexible Environment for TensorFlow Photo Analysis

To use this demonstration, the TensorFlow Object Detection API should be used. For more details [here](https://github.com/tensorflow/models/tree/master/object_detection).

To use this example you will need these minimum elements of [TensorFlow Object Detection API](https://github.com/tensorflow/models/tree/master/object_detection):

Object model, quick option to automatize the deploy:

``` 
git clone https://github.com/tensorflow/models.git
```
*validate dependency with protoc tool

```
protoc ./models/research/object_detection/protos/string_int_label_map.proto --python_out=.
```

```
cp -R models/research/object_detection/ object_detection/
```
```
rm -rf model
```

You can use the preferred model: faster_rcnn_inception_resnet_v2_atrous_coco_2017_11_08 or as another like faster_rcnn_inception_v2_coco_2017_11_08


# Deploy the project

Note: You should have the [Google Cloud SDK](https://cloud.google.com/sdk/docs/). More information about App Engine Flexible environment, Python [here](https://cloud.google.com/appengine/docs/flexible/python/quickstart)

Local:
python main.py

Production Environment:
gcloud app deploy 

*(-v version) if you want to deploy it to a specific version.