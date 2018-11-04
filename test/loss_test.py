import GeoSimilarity.src.inputs as ginput
import GeoSimilarity.src.model as gmodel
import GeoSimilarity.src.loss as gloss
import tensorflow as tf


class TestLoss(tf.test.TestCase):
    def test_loss(self):
        batch_size = 64
        img_size = 64
        channels = 4

        input = ginput.Input()
        input_a, input_b = input.get_batch(batch_size, img_size, channels)
        labels = tf.cast(tf.random_uniform([batch_size, 1], minval=0, maxval=2, dtype=tf.int32), dtype=tf.float32)

        model = gmodel.Model(model_type='early_fusion', is_train=True, grid_size=1)
        model.build(img_a=input_a, img_b=input_b, labels=labels)

        # Define Loss
        loss_fn = gloss.Loss(model=model)

        loss = loss_fn.compute_loss()

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            np_loss = sess.run(loss)

            assert np_loss.shape == ()
            print('Loss value: %s' % np_loss)