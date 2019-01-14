import tensorflow as tf
import GeoSimilarity.src.sampling as sampling


class TestSampling(tf.test.TestCase):

    def test_extract_neighbor(self):
        print('Test neighbor extraction')
        n_channels = 4
        n_frames = 12
        img_size = 1024
        patch_size = 64
        channels_frames = n_channels * n_frames
        neighborhood_factor = 2

        data = tf.random_normal(shape=(img_size, img_size, channels_frames))

        neighborhood, index = sampling.extract_neighborhood(data, img_size, patch_size, channels_frames, neighborhood_factor)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(10):
                np_data, np_neighborhood, np_index = sess.run([data, neighborhood, index])
                print(np_index)

                assert (np_neighborhood - np_data[np_index[0]:np_index[0] + patch_size * neighborhood_factor,
                                          np_index[1]:np_index[1] + patch_size * neighborhood_factor, :]).sum() == 0

    def test_extract_time_series(self):
        print('Test time series extraction')
        n_channels = 4
        n_frames = 12
        img_size = 1024
        patch_size = 64
        channels_frames = n_channels * n_frames
        neighborhood_factor = 2

        data = tf.random_normal(shape=(img_size, img_size, channels_frames))

        neighborhood, index = sampling.extract_neighborhood(data,
                                                            img_size,
                                                            patch_size,
                                                            channels_frames,
                                                            neighborhood_factor)

        time_series, neighbor_time_series, new_index0, new_index1 = sampling.extract_time_series(neighborhood,
                                                                                                 patch_size,
                                                                                                 channels_frames,
                                                                                                 neighborhood_factor)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(10):
                np_neighborhood, np_time_series, np_neighbor_time_series,\
                np_new_index0, np_new_index1 = sess.run([neighborhood,
                                                         time_series,
                                                         neighbor_time_series,
                                                         new_index0,
                                                         new_index1])
                print(np_new_index0)
                print(np_new_index1)

                assert (np_time_series - np_neighborhood[np_new_index0[0] * patch_size:(np_new_index0[0] + 1) * patch_size,
                                          np_new_index0[1] * patch_size:(np_new_index0[1] + 1) * patch_size, :]).sum() == 0

                assert (np_neighbor_time_series - np_neighborhood[np_new_index1[0] * patch_size:(np_new_index1[0] + 1) * patch_size,
                                                  np_new_index1[1] * patch_size:(np_new_index1[1] + 1) * patch_size, :]).sum() == 0

    def test_extract_imgs(self):
        print('Test image extraction')
        n_channels = 4
        n_frames = 12
        img_size = 1024
        patch_size = 64
        channels_frames = n_channels * n_frames
        neighborhood_factor = 2

        data = tf.random_normal(shape=(img_size, img_size, channels_frames))

        neighborhood, index = sampling.extract_neighborhood(data,
                                                            img_size,
                                                            patch_size,
                                                            channels_frames,
                                                            neighborhood_factor)

        time_series, neighbor_time_series, new_index0, new_index1 = sampling.extract_time_series(neighborhood,
                                                                                                 patch_size,
                                                                                                 channels_frames,
                                                                                                 neighborhood_factor)
        image_o, image_f, index_o, index_f = sampling.extract_same_location_images(time_series, n_channels, n_frames, patch_size)
        image_t = sampling.extract_same_time_image(neighbor_time_series, n_channels, patch_size, index_o)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(10):
                np_time_series, np_neighbor_time_series, np_image_o,\
                np_image_f, np_image_t, np_index_o, np_index_f = sess.run([time_series,
                                                                           neighbor_time_series,
                                                                           image_o,
                                                                           image_f,
                                                                           image_t,
                                                                           index_o,
                                                                           index_f])
                print(np_index_o)
                print(np_index_f)

                assert (np_image_o - np_time_series[:, :, n_channels * np_index_o:n_channels * (np_index_o + 1)]).sum() == 0
                assert (np_image_f - np_time_series[:, :, n_channels * np_index_f:n_channels * (np_index_f + 1)]).sum() == 0
                assert (np_image_t - np_neighbor_time_series[:, :, n_channels * np_index_o:n_channels * (np_index_o + 1)]).sum() == 0

    def test_extract_tuples(self):
        print('Test tuple extraction')
        batch_size = 2
        n_channels = 4
        n_frames = 12
        img_size = 1024
        patch_size = 64
        channels_frames = n_channels * n_frames
        neighborhood_factor = 2
        n_patches = 64

        inputs = tf.random_normal(shape=(batch_size, img_size, img_size, channels_frames))

        reference_batch, same_location_batch, same_time_batch = sampling.extract_tuples(inputs, n_patches, img_size,
                                                                                        patch_size, n_channels, neighborhood_factor)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(10):
                np_reference_batch, np_same_location_batch, np_same_time_batch = sess.run([reference_batch,
                                                                                           same_location_batch,
                                                                                           same_time_batch])
                print(np_reference_batch.shape)
                print(np_same_location_batch.shape)
                print(np_same_time_batch.shape)