import tensorflow as tf
from tensorflow.python.summary import summary


def compute_weight_decay(var_list, coeff):
    weight_norm = coeff * tf.reduce_sum(tf.stack([tf.nn.l2_loss(var) for var in var_list]))
    return weight_norm


class Loss(object):
    def __init__(self, model, weight_decay_coeff=0.0):
        self.model = model
        self.weight_decay_coeff = weight_decay_coeff

    def __call__(self):
        self.loss = tf.reduce_mean(tf.squared_difference(self.model.labels, self.model.outputs))

        if self.weight_decay_coeff > 0.0:
            weight_decay_loss = compute_weight_decay(self.model.var_list, self.weight_decay_coeff)
            summary.scalar('weight_decay', weight_decay_loss)
            self.loss = self.loss + weight_decay_loss

        summary.scalar('loss', self.loss)
        return self.loss

    def compute_loss(self):
        return self.__call__()
