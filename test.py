import tensorflow as tf
with tf.device("/gpu:0"):
    y = tf.scatter_nd([[1], [2], [3]], [1., 1., 1.], (5,))

session_config = tf.ConfigProto(allow_soft_placement=False,
                                  log_device_placement=True)

with tf.Session(config=session_config) as sess:
    print(sess.run(y))
