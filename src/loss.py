import tensorflow as tf
from tensorflow.python.summary import summary


class Loss(object):
    def __init__(self, model):
        self.model = model

    def __call__(self):
        self.loss = tf.reduce_mean(tf.squared_difference(self.model.labels, self.model.outputs))
        summary.scalar('loss', self.loss)
        return self.loss

    def compute_loss(self):
        return self.__call__()
