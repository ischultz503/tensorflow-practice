#mple for learning a regression."""


import tensorflow as tf
import numpy

# Parameters
learning_rate = 0.01
training_epochs = 300
display_step = 50

# Generate training data
train_X = []
train_Y = []
f = lambda x: x**2
for x in range(-20, 20):
    train_X.append(float(x))
    train_Y.append(f(x))
train_X = numpy.asarray(train_X)
train_Y = numpy.asarray(train_Y)
n_samples = train_X.shape[0]

# Graph input
X = tf.placeholder(tf.float32)
reshaped_X=tf.reshape(X,[-1,1])
Y = tf.placeholder(tf.float32)

# Create Model
W1 = tf.Variable(tf.truncated_normal([1, 10], stddev=0.1), name="weight")
b1 = tf.Variable(tf.constant(0.1, shape=[1, 10]), name="bias")
mul =tf.matmul(reshaped_X, W1)
h1 = tf.nn.sigmoid(mul + b1)
W2 = tf.Variable(tf.truncated_normal([10, 1], stddev=0.1), name="weight")
b2 = tf.Variable(tf.constant(0.1, shape=[1]), name="bias")
activation = tf.nn.sigmoid(tf.matmul(h1, W2) + b2)

# Minimize the squared errors
l2_loss = tf.reduce_sum(tf.pow(activation-Y, 2))/(2*n_samples)
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(l2_loss)

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Fit all training data
    for epoch in range(training_epochs):
        for (x, y) in zip(train_X, train_Y):
            sess.run(optimizer, feed_dict={X: x, Y: y})

        # Display logs per epoch step
        if epoch % display_step == 0:
            cost = sess.run(l2_loss, feed_dict={X: train_X, Y: train_Y})
            print("Epoch: {:04d}, cost={:.9f}".format((epoch+1), cost),
                  "W=", sess.run(W1), "b=", sess.run(b1))

    print("Optimization Finished!")
    print("cost=", sess.run(l2_loss, feed_dict={X: train_X, Y: train_Y}),
          "W1=", sess.run(W1), "b2=", sess.run(b2))
    print("W2=", sess.run(W2))
    print("b2=", sess.run(b2))
    print("b1=",sess.run(b1))

