import logging
import os
import sys

import tensorflow as tf

#a = tf.placeholder("float")
#b = tf.placeholder("float")

a = tf.Variable(1,name="a")
b = tf.Variable(1,name="b")

y = tf.multiply(a, b)

#init_op = tf.global_variables_initializer()

saver = tf.train.Saver()

with tf.Session() as sess:
    result = str(sess.run(y))
    #sess.run(init_op)
    saver.save(sess, 'my-model.ckpt')