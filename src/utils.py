from tensorflow.python.training import session_run_hook
from tensorflow.python.summary import summary
import tensorflow as tf
import os


def create_folder(folder_list):
    """
    Create folders if they have not been created
    """
    for folder in folder_list:
        if not tf.gfile.Exists(folder):
            tf.gfile.MakeDirs(folder)


def load_model(session, saver, model_dir):
    """
    This method restores the model written in a checkpoint file.
    A tensorflow session must be initialized before ise this function

    :param session: tf.Session() object
    :param saver: tf.train.Saver() object
    :param model_dir: directory to the checkpoint files (.ckpt)

    """
    ckpt = tf.train.get_checkpoint_state(model_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        saver.restore(session, os.path.join(model_dir, ckpt_name))
        print("Model successfully restored: %s" % ckpt_name)
    else:
        print("Model cannot be restored")


def create_global_step():
    """
    Create global step
    """
    global_step = tf.train.get_or_create_global_step(graph=None)
    global_step_inc = global_step.assign_add(1)
    summary.scalar('global_steps', global_step)
    return global_step_inc


class RunTrainOpsHook(session_run_hook.SessionRunHook):
    """
    A hook class to run train ops a fixed number of times.
    """

    def __init__(self, train_ops, train_steps=1):
        """
        Run train ops a certain number of times.

        :param train_ops: A train op or iterable of train ops to run.
        :param train_steps: The number of times to run the op(s).
        """
        if not isinstance(train_ops, (list, tuple)):
            train_ops = [train_ops]
        self._train_ops = train_ops
        self._train_steps = train_steps

    def before_run(self, run_context):
        for _ in range(self._train_steps):
            run_context.session.run(self._train_ops)


