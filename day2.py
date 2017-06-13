'''
Day 2: Continuing my Tensor Flow Journey
=========================================
It has been days since I continued my Tensor Flow journey
So I am continuing now. 

Topics
======
Contrib learn

'''



import tensorflow as tf
import os
import numpy as np

#Disabling SESS build warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


#Features list
features = [tf.contrib.layers.real_valued_column("x",dimension=1)]

#Estimator

estimator = tf.contrib.learn.LinearRegressor(feature_columns=features)


#Tensorflow data
x = np.array([1.,2.,3.,4.])
y = np.array([0.,-1.,-2.,-3.])
input_fn = tf.contrib.learn.io.numpy_input_fn({"x":x},y,batch_size=4,num_epochs=1000)


#Training data set 
estimator.fit(input_fn=input_fn,steps=1000)

print(estimator.evaluate(input_fn=input_fn))
