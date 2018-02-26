import tensorflow as tf

constant_x = tf.constant(5, name='constant_x')
variable_y = tf.Variable(constant_x + 5, name='variable_y')
print (variable_y)

#initialize all variables
init = tf.global_variables_initializer()
# All variables are now initialized

with tf.Session() as sess:
    sess.run(init)
    print(sess.run(variable_y))
