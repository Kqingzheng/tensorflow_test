# import tensorflow as tf
#
# tf.compat.v1.disable_eager_execution()
# hello = tf.constant('hello,tensorflow')
# sess = tf.compat.v1.Session()
# print(sess.run(hello))
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())