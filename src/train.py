""""
Train the model
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import GeoSimilarity.src.inputs as ginput
import GeoSimilarity.src.model as gmodel
import GeoSimilarity.src.loss as gloss
import GeoSimilarity.src.train_op as gtrain_op
import GeoSimilarity.src.utils as utils

import tensorflow as tf

from tensorflow.contrib.training.python.training import training

flags = tf.flags

# Data parameters
flags.DEFINE_integer('img_size', 64, 'Image height and width.')
flags.DEFINE_integer('channels', 4, 'The number of channels in the raw image.')
flags.DEFINE_integer('batch_size', 2, 'The number of images in each batch.')

flags.DEFINE_integer('grid_size', 1, 'Grid size to visualize generated and real images.')
flags.DEFINE_string('model_type', 'late_fusion', 'Model type can be early_fusion or late_fusion .')

# Model and dataset directories
flags.DEFINE_string('train_log_dir',
                    '/home/hugs/Models/mymodel',
                    'Directory to write event logs and save the model.')

flags.DEFINE_string('dataset_dir',
                    '/home/hugs/Data',
                    'Directory containing data from one region of interest.')

flags.DEFINE_string('dataset_name',
                    'peron.tfrecord',
                    'Name of the tfrecord dataset.')

flags.DEFINE_integer('max_number_of_steps', 150000, 'The maximum number of gradient steps.')
flags.DEFINE_integer('save_checkpoint_secs', 300, 'The required time to save a checkpoint.')
flags.DEFINE_integer('save_summaries_steps', 30, 'The step number to save summaries.')
flags.DEFINE_float('learning_rate', 0.0002, 'The initial learning rate.')
flags.DEFINE_float('beta', 0.5, 'The beta value for the Adam optimizer.')
flags.DEFINE_integer('decay_steps', 50000, 'The decay steps for the learning rate.')
flags.DEFINE_bool('staircase', True, 'Learning rate shape.')

FLAGS = flags.FLAGS


def main(_):
    utils.create_folder(folder_list=[FLAGS.dataset_dir, FLAGS.train_log_dir])

    # Force all input processing onto CPU in order to reserve the GPU for
    # the forward inference and back-propagation.
    with tf.name_scope('inputs'):
        with tf.device('/cpu:0'):
            input = ginput.Input()
            input_a, input_b = input.get_batch(FLAGS.batch_size, FLAGS.img_size, FLAGS.channels)
            labels = tf.cast(tf.random_uniform([FLAGS.batch_size, 1], minval=0, maxval=2, dtype=tf.int32), dtype=tf.float32)

    # Define the Model
    model = gmodel.Model(model_type=FLAGS.model_type, is_train=True, grid_size=FLAGS.grid_size)
    model.build(img_a=input_a, img_b=input_b, labels=labels)

    # Define Loss
    loss_fn = gloss.Loss(model=model)

    # Get global step and the respective operation
    global_step_inc = utils.create_global_step()

    # Create training operation
    train_op = gtrain_op.TrainOp(model=model,
                                 loss_fn=loss_fn,
                                 learning_rate=FLAGS.learning_rate,
                                 decay_steps=FLAGS.decay_steps,
                                 beta=FLAGS.beta,
                                 staircase=FLAGS.staircase)

    op = train_op.create_train_op()

    status_message = tf.string_join(['Starting train step: ', tf.as_string(tf.train.get_or_create_global_step())],
                                    name='status_message')

    hooks = [tf.train.LoggingTensorHook([status_message], every_n_iter=10),
             tf.train.StopAtStepHook(num_steps=FLAGS.max_number_of_steps),
             utils.RunTrainOpsHook(train_ops=op)]

    training.train(train_op=global_step_inc,
                   logdir=FLAGS.train_log_dir,
                   master='',
                   is_chief=True,
                   scaffold=None,
                   hooks=hooks,
                   chief_only_hooks=None,
                   save_checkpoint_secs=FLAGS.save_checkpoint_secs,
                   save_summaries_steps=FLAGS.save_summaries_steps,
                   config=None,
                   max_wait_secs=7200)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
