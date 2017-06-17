from tensorflow.examples.tutorials.mnist import input_data

#GEtting the training example
mnist = input_data.read_data_sets("MNSIT_data/",one_hot=True)

import tensorflow as tf


#Defining the model 
'''
Each picture is 28 * 28 pixels 
Training set has picture,label format
model is for each pixel and a weight attached we find the W vector that corresponds
to the training examples
Using softmare we get an array [10] with probabilities b/w 0-1
'''
x = tf.placeholder(tf.float32,[None,784]) #Meaning 2d array of [any number, 784(pixels in picture)]
W = tf.Variable(tf.zeros([784,10])) #W.x -> [10,1] matrix
b = tf.Variable(tf.zeros([10])) # Adding bais to W.x
y = tf.matmul(x,W)+b

y_ = tf.placeholder(tf.float32,[None,10])
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
for _ in range(1000):
	batch_xs, batch_ys = mnist.train.next_batch(100)
	sess.run(train_step,feed_dict={x: batch_xs,y_:batch_ys})


correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
print(sess.run(accuracy,feed_dict={x: mnist.test.images, y_: mnist.test.labels}))


