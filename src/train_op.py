import tensorflow as tf
from tensorflow.python.summary import summary


class TrainOp(object):
    """
    Create a training operation
    """
    def __init__(self, model, loss_fn, learning_rate, decay_steps, beta, staircase):
        self.model = model
        self.loss_fn = loss_fn
        self.learning_rate = learning_rate
        self.decay_steps = decay_steps
        self.beta = beta
        self.staircase = staircase

    def create_train_op(self):
        # When training, the moving_mean and moving_variance need to be updated (tf.layer.batch_norm)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            # Get the learning rate
            lr = tf.train.exponential_decay(learning_rate=self.learning_rate,
                                            global_step=tf.train.get_or_create_global_step(),
                                            decay_steps=self.decay_steps,
                                            decay_rate=0.9,
                                            staircase=self.staircase)
            summary.scalar('learning_rate', lr)

            optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=self.beta)

            train_op = optimizer.minimize(loss=self.loss_fn.compute_loss(), var_list=self.model.var_list)

            return train_op
