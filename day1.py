'''
Day 1: Tensor flow tutorial
============================
Following the google tensorflow tutorial I am dividing it into days
based on how I learnt

Things learnt day1:
Tensors
Constants
Placeholder
Variables
Sessions
Linear model
Gradient descent minimizer
'''

import tensorflow as tf
import os

#Disabling SESS build warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


#Constant addition
node1 = tf.constant(3.0,tf.float32)
node2 = tf.constant(4.0)

sess = tf.Session()

print(sess.run([node1,node2]))


#Placeholder addition and other ops
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b
print("adder node 1: ", sess.run(adder_node,{a:1,b:3}))
print("adder node 2: ",sess.run(adder_node,{a:[1,5],b:[3,10]}))
adder_triple_node = adder_node * 3 #Try adder_node ** 3
print("adder triple node 1: ",sess.run(adder_triple_node,{a:1,b:3}))
print("adder triple node 2: ",sess.run(adder_triple_node,{a:[1,5],b:[3,10]}))

#Liner model
W = tf.Variable([0.3],tf.float32)
b = tf.Variable([-0.3],tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W*x + b
init = tf.global_variables_initializer()
sess.run(init)
print("Linear model test: ",sess.run(linear_model,{x:[1,2,3,4]}))
#loss function
y = tf.placeholder(tf.float32)
sq_delta = tf.square(linear_model-y)
loss = tf.reduce_sum(sq_delta)
print("Loss with input data [0,-1,-2,-3]: ",sess.run(loss,{x:[1,2,3,4],y:[0,-1,-2,-3]}))
#We fix W and b for making loss = 0
fixW = tf.assign(W,[-1])
fixb = tf.assign(b,[1])
sess.run([fixW,fixb])
print("Loss with input data [0,-1,-2,-3]: ",sess.run(loss,{x:[1,2,3,4],y:[0,-1,-2,-3]}))

#Machine learning using gradient descent
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)
sess.run(init)
for i in range(1,1000):
	sess.run(train,{x:[1,2,3,4],y:[0,-1,-2,-3]})
print("Learnt values of W and b: ",sess.run([W,b]))

