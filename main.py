import logging
import os
import sys

import numpy as np
import tensorflow as tf
from gcloud import storage

from flask import Flask


app = Flask(__name__)



@app.route('/')
def hello():

    m = 50
    n = 1

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

    train_Y = np.array([
        [7.7918926e-01],
        [9.1596757e-01],
        [9.0538354e-01],
        [9.0566138e-01],
        [9.3898890e-01],
        [9.6684740e-01],
        [9.6436824e-01],
        [9.1445939e-01],
        [9.3933944e-01],
        [9.6074971e-01],
        [8.9837094e-01],
        [9.1209739e-01],
        [9.4238499e-01],
        [9.6624578e-01],
        [1.0526500e+00],
        [1.0143791e+00],
        [9.5969426e-01],
        [9.6853716e-01],
        [1.0766065e+00],
        [1.1454978e+00],
        [1.0340625e+00],
        [1.0070009e+00],
        [9.6683648e-01],
        [1.0895919e+00],
        [1.0634462e+00],
        [1.1237239e+00],
        [1.0323374e+00],
        [1.0874452e+00],
        [1.0702988e+00],
        [1.1606493e+00],
        [1.0778037e+00],
        [1.1069758e+00],
        [1.0971875e+00],
        [1.1648603e+00],
        [1.1411796e+00],
        [1.0844156e+00],
        [1.1252493e+00],
        [1.1168341e+00],
        [1.1970789e+00],
        [1.2069462e+00],
        [1.1251046e+00],
        [1.1235672e+00],
        [1.2132829e+00],
        [1.2522652e+00],
        [1.2497065e+00],
        [1.1799706e+00],
        [1.1897299e+00],
        [1.3029934e+00],
        [1.2601134e+00],
        [1.2562267e+00]

    ]
    ).astype('float32')


    mean = np.mean(train_X, axis=0)
    std = np.std(train_X, axis=0)

    train_X_2 = np.nan_to_num((train_X - mean) / std)

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

    X = tf.placeholder(tf.float32, [m, n])
    Y = tf.placeholder(tf.float32, [m, 1])

    # weight and bias
    W = tf.Variable(tf.zeros([n, 1], dtype=np.float32), name="weight")
    b = tf.Variable(tf.zeros([1], dtype=np.float32), name="bias")

    # linear model
    with tf.name_scope("linear_Wx_b") as scope:
        activation = tf.add(tf.matmul(X, W), b)

    # cost
    with tf.name_scope("cost") as scope:
        cost = tf.reduce_sum(tf.square(activation - Y)) / (2 * m)
        tf.summary.scalar("cost", cost)

    # train
    with tf.name_scope("train") as scope:
        optimizer = tf.train.GradientDescentOptimizer(0.07).minimize(cost)

    #saver = tf.train.Saver()

    # tensorflow session
    with tf.Session() as sess:

        #tf.initialize_all_variables().run()



        client = storage.Client('lookthiscode-521')
        bucket = client.bucket('models_lookthiscode')
        blob = bucket.blob('my-model.ckpt.meta')

        with open('my-model.ckpt.meta', 'w') as f:
            blob.download_to_file(f)

        blob1 = bucket.blob('checkpoint')
        with open('checkpoint', 'w') as f:
            blob1.download_to_file(f)

        blob2 = bucket.blob('my-model.ckpt.index')
        with open('my-model.ckpt.index', 'w') as f:
            blob2.download_to_file(f)

        blob3 = bucket.blob('my-model.ckpt.data-00000-of-00001')
        with open('my-model.ckpt.data-00000-of-00001', 'w') as f:
            blob3.download_to_file(f)

        #saver.restore(sess, "/tmp/my-model.meta")

        saver = tf.train.import_meta_graph('my-model.ckpt.meta')
        saver.restore(sess, tf.train.latest_checkpoint('./'))

        init = tf.global_variables_initializer()
        sess.run(init)

        training_cost = sess.run(cost, feed_dict={X: np.asarray(train_X_2), Y: np.asarray(train_Y)})

        predict_X_2 = np.array([7], dtype=np.float32).reshape([1, 1])


        predict_X_2 = (predict_X_2 - mean) / std
        logging.info(predict_X_2)
        predict_Y_2 = tf.add(tf.matmul(predict_X_2, W), b)
        logging.info(predict_Y_2)

        return str(sess.run(predict_Y_2))


@app.errorhandler(500)
def server_error(e):
    logging.exception('An error occurred during a request.')
    return """
    An internal error occurred: <pre>{}</pre>
    See logs for full stacktrace.
    """.format(e), 500


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080, debug=True)
