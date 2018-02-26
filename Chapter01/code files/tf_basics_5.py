import tensorflow as tf

x = tf.placeholder("float", [None, 3])
y = x * 2

with tf.Session() as session:
    input_data = [[1, 2, 3],
                 [4, 5, 6],]
    result = session.run(y, feed_dict={x: input_data})
    print(result)
