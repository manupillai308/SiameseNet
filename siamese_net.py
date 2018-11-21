#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf

tf.reset_default_graph()

def he_initializer(shape):
    return tf.contrib.layers.variance_scaling_initializer()(shape)

alpha = 0.1
learning_rate= 0.01
epochs = 250

with tf.variable_scope('input'):
    anchor = tf.placeholder(shape=[None, 28, 28, 1], dtype=tf.float32)
    positive = tf.placeholder(shape=[None, 28, 28, 1], dtype=tf.float32)
    negative = tf.placeholder(shape=[None, 28, 28, 1], dtype=tf.float32)
    
with tf.variable_scope('network-weights'):
    k1 = tf.Variable(he_initializer([5, 5, 1, 16]))
    b1 = tf.Variable(tf.zeros([16]))
    
    k2 = tf.Variable(he_initializer([3, 3, 16, 32]))
    b2 = tf.Variable(tf.zeros([32]))
    
    W1 = tf.Variable(he_initializer([7 * 7 * 32, 128]))
    b3 = tf.Variable(tf.zeros([128]))
    
    W2 = tf.Variable(he_initializer([128,10]))
    b4 = tf.Variable(tf.zeros([10]))
    
def create_network(inp):
    #reshape1= tf.reshape(inp, [-1, 28, 28, 1])
    conv1 = tf.nn.relu(tf.nn.conv2d(inp, filter=k1, strides=[1,2,2,1], padding="SAME") + b1)
    conv2 = tf.nn.relu(tf.nn.conv2d(conv1, filter=k2, strides=[1,2,2,1], padding="SAME") + b2)
    reshape2 = tf.reshape(conv2, [-1, 7 * 7 * 32])
    dense1 = tf.nn.relu(tf.matmul(reshape2, W1)+ b3)
    logits = tf.nn.sigmoid(tf.matmul(dense1, W2) + b4)
    return logits

with tf.variable_scope('networks'):
    net1 = create_network(anchor)
    net2 = create_network(positive)
    net3 = create_network(negative)
    
    positive_loss = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(net1-net2), axis=1)))
    negative_loss = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(net1-net3), axis=1)))
    loss = tf.maximum(positive_loss-negative_loss+alpha, 0) #alpha is known as 'margin', added to avoid solution converging to embeddings=0 for all x.
    optimizer = tf.train.MomentumOptimizer(momentum=0.99, use_nesterov=True, learning_rate=learning_rate)
    training_op = optimizer.minimize(loss)
    
init = tf.global_variables_initializer()
#anchor_im = None
#neg_im = None
#pos_im = None
with tf.Session(config=tf.ConfigProto(allow_soft_placement=True) as sess:
	init.run()
	for i in range(epochs):
		#sess.run(training_op, feed_dict={anchor:anchor_im, positive:pos_im, negative:neg_im})
	
