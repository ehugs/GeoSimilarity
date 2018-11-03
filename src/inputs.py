import tensorflow as tf


class Input(object):
    """
    Fake inputs for testing purpose
    """
    def __init__(self):
        pass

    def get_batch(self, batch_size, img_size, channels):
        input_a = tf.random_normal(mean=0.0, stddev=1.0, shape=[batch_size, img_size, img_size, channels])
        input_b = tf.random_normal(mean=0.0, stddev=1.0, shape=[batch_size, img_size, img_size, channels])

        return input_a, input_b


if __name__ == '__main__':
    input = Input()
    input_a, input_b = input.get_batch(64, 64, 3)

    sess = tf.Session()
    print(sess.run(input_a).mean())
    print(sess.run(input_b).mean())