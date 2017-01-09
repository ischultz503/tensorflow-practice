#notes
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

import tensorflow as tf

#Parameters
learning_rate=0.001
training_epochs=15
batch_size=100
display_step=1

#Network parameters
n_hidden_1 = 256
n_hidden_2 = 256
n_input = 784
n_classes = 10

#tf graph input
x=tf.placeholder("float", [None, n_input])
y=tf.placeholder("float", [None, n_classes])

#Create model
def multilayer_perceptron(x, weights, biases):
	layer_1=tf.nn.relu(tf.add(tf.matmul(x, weights['w1']), biases['b1']))
	layer_2=tf.nn.relu(tf.add(tf.matmul(layer_1, weights['w2']), biases['b2']))
	out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
	return out_layer

#Store layer weights and biases
weights = {
	'w1' : tf.Variable(tf.random_normal([n_input, n_hidden_1])),
	'w2' : tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
	'out' : tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
	'b1' : tf.Variable(tf.random_normal([n_hidden_1])),
	'b2' : tf.Variable(tf.random_normal([n_hidden_2])),
	'out' : tf.Variable(tf.random_normal([n_classes]))
}

#Construct model
pred=multilayer_perceptron(x, weights, biases)

#Define loss(softmax) and optimizer
cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

#Initialize all variables
init = tf.initialize_all_variables()

#Launch the GRAPH!!!!
with tf.Session() as sess:
	sess.run(init)

	#Training cycle
	for epoch in range(training_epochs):
		avg_cost =0.
		total_batch = int(mnist.train.num_examples/batch_size)
		#Loop over all of the batches
		for i in range(total_batch):
			batch_xs, batch_ys = mnist.train.next_batch(batch_size)
			# Fit training using batch data
			sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})
			#Compute average loss
			avg_cost += sess.run(cost, feed_dict={x: batch_xs, y: batch_ys})/total_batch
			tf.summary.scalar('avg_cost', avg_cost)
			tf.summary.histogram('avg_cost',avg_cost)
		#Display logs perepoch step
		if epoch % display_step == 0:
			print ("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))
	
	print ("Optimiziation finished.")

#Test model
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
#Calculate accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
with tf.Session() as sess:
	sess.run(init)
	print ("Accuracy:", accuracy.eval({x: mnist.test.images, y:mnist.test.labels}))
