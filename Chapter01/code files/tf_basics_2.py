import tensorflow as tf

# A is an int32 tensor with rank = 0
A = tf.constant(123) 
# B is an int32 tensor with dimension of 1 ( rank = 1 ) 
B = tf.constant([123,456,789]) 
# C is an int32 2- dimensional tensor 
C = tf.constant([ [123,456,789], [222,333,444] ])

#Creating a session object for execution of the computational graph
with tf.Session() as sess:
    #Implementing the tf.constant operation in the session
    output = sess.run(B)
    print(output)
