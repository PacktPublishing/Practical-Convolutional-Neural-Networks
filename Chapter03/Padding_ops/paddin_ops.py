import tensorflow as tf
import warnings
import os

from tensorflow.python.framework import ops

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
ops.reset_default_graph()

X = tf.constant([[1., 2., 3.],
                 [4., 5., 6.]])

X_reshaped = tf.reshape(X, [1, 2, 3, 1])  # give a shape accepted by tf.nn.max_pool

VALID = tf.nn.max_pool(X_reshaped, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID')
SAME = tf.nn.max_pool(X_reshaped, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

print(VALID.get_shape()) #== [1, 1, 1, 1]  # valid_pad is [5.]
print(SAME.get_shape()) #== [1, 1, 2, 1]   # same_pad is  [5., 6.]
