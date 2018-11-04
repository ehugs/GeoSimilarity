import tensorflow as tf


class Loss(object):
    def __init__(self, model):
        self.model = model

    def __call__(self):
        self.loss = tf.reduce_mean(tf.squared_difference(self.model.labels, self.model.outputs))
        return self.loss

    def compute_loss(self):
        return self.__call__()
