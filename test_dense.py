import unittest
import tensorflow as tf

class MyTestCase(unittest.TestCase):
    def test_something(self):
        input = tf.placeholder(dtype=tf.float32,shape=[1,2])
        a1 = [[1.0,1.0]]
        a2 = [[2.0,2.0]]
        a3 = [[10.0,10.0]]
        b = tf.layers.dense(input,10)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            print sess.run(b,feed_dict={input:a1})
            print sess.run(b,feed_dict={input:a2})
            print sess.run(b,feed_dict={input:a3})


if __name__ == '__main__':
    unittest.main()
