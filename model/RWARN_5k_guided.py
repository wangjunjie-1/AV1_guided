import tensorflow as tf
import numpy as np


def guided_model_5k(input_tensor):
    conv_w0=tf.get_variable("conv_w0",[3,3,1,12],initializer=tf.contrib.layers.xavier_initializer())
    conv_b0=tf.get_variable("conv_b0",[12],initializer=tf.constant_initializer(0))
    tf.add_to_collection(tf.GraphKeys.WEIGHTS,tf.contrib.layers.l2_regularizer(1.)(conv_w0))

    tensor=tf.nn.bias_add(tf.nn.conv2d(input_tensor,conv_w0,strides=[1,1,1,1],padding='SAME'),conv_b0)
    convId=0
    for i in range(3):
        tensor=ResBlock(tensor,"conv_layer%02d"%(convId),times=1)
        convId+=1

    conv_w1 = tf.get_variable("conv_w1", [3, 3, 12, 2], initializer=tf.contrib.layers.xavier_initializer())
    conv_b1 = tf.get_variable("conv_b1", [2], initializer=tf.constant_initializer(0))
    tf.add_to_collection(tf.GraphKeys.WEIGHTS, tf.contrib.layers.l2_regularizer(1.)(conv_w0))

    tensor = tf.nn.bias_add(tf.nn.conv2d(tensor, conv_w1, strides=[1, 1, 1, 1], padding='SAME'), conv_b1)

    tensor = tf.add(tensor, input_tensor)
    return tensor

def ResBlock(temp_tensor,name,times=1):
    skip_tensor = temp_tensor
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        # ---------------------------------------------------------------------------------------------------------------------------------
        # Conv, 1x1, filters=32 ,+ ReLU
        conv_w1 = tf.get_variable("conv_w1", [1, 1, 12, 32],
                                  initializer=tf.contrib.layers.xavier_initializer())
        conv_b1 = tf.get_variable("conv_b1", [32], initializer=tf.constant_initializer(0))
        tf.add_to_collection(tf.GraphKeys.WEIGHTS, tf.contrib.layers.l2_regularizer(1.)(conv_w1))
        # ---------------------------------------------------------------------------------------------------------------------------------
        # Conv, 1x1, filters=8
        conv_w2 = tf.get_variable("conv_w2", [1, 1, 32, 8],
                                  initializer=tf.contrib.layers.xavier_initializer())
        conv_b2 = tf.get_variable("conv_b2", [8], initializer=tf.constant_initializer(0))
        tf.add_to_collection(tf.GraphKeys.WEIGHTS, tf.contrib.layers.l2_regularizer(1.)(conv_w2))
        # ---------------------------------------------------------------------------------------------------------------------------------
        # Conv, 3x3, filters=12
        conv_w3 = tf.get_variable("conv_w3", [3, 3, 8, 12],
                                  initializer=tf.contrib.layers.xavier_initializer())
        conv_b3 = tf.get_variable("conv_b3", [12], initializer=tf.constant_initializer(0))
        tf.add_to_collection(tf.GraphKeys.WEIGHTS, tf.contrib.layers.l2_regularizer(1.)(conv_w3))
        # ---------------------------------------------------------------------------------------------------------------------------------
    for i in range(times):
        temp_tensor = tf.nn.relu(
            tf.nn.bias_add(tf.nn.conv2d(temp_tensor, conv_w1, strides=[1, 1, 1, 1], padding='SAME'), conv_b1))
        temp_tensor = tf.nn.bias_add(tf.nn.conv2d(temp_tensor, conv_w2, strides=[1, 1, 1, 1], padding='SAME'), conv_b2)
        temp_tensor = tf.nn.bias_add(tf.nn.conv2d(temp_tensor, conv_w3, strides=[1, 1, 1, 1], padding='SAME'), conv_b3)
    # ---------------------------------------------------------------------------------------------------------------------------------
    # skip + out_tensor
    out_tensor = tf.add(skip_tensor, temp_tensor)
    return out_tensor
