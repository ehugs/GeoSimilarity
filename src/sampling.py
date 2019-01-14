"""
Contains the code to get same-location images from a time series provided by the data_provided function.
"""

import tensorflow as tf


def extract_same_location_images(time_series, n_channels, n_frames, patch_size):
    """
    This method extracts two images from a time series

    :param time_series: A tensor of shape [patch_size, patch_size, channels * frames]
    :param n_channels: Number of channels for each image
    :param n_frames: Number of frames for each time series
    :param patch_size: Patch size

    :return: Two tensors of shape [patch_size, patch_size, channels] each one and the time index
    """

    # Get two random time index for image spatial location
    time_index = tf.random_uniform(shape=[2], minval=0, maxval=n_frames, dtype=tf.int32)

    # Get the corresponding two images
    index_o = time_index[0]
    index_f = time_index[1]

    image_o = time_series[:, :, index_o * n_channels: (index_o + 1) * n_channels]
    image_f = time_series[:, :, index_f * n_channels: (index_f + 1) * n_channels]

    image_o.set_shape([patch_size, patch_size, n_channels])
    image_f.set_shape([patch_size, patch_size, n_channels])

    return image_o, image_f, index_o, index_f


def extract_same_time_image(neighbor_time_series, n_channels, patch_size, index_o):
    """
    This method extracts an image from a different location but acquired at the same time w.r.t time_series

    :param neighbor_time_series: A tensor of shape [patch_size, patch_size, channels * frames]
    :param n_channels: Number of channels for each image
    :param index_o: First time index of time_series
    :param patch_size: Patch size

    :return: A tensor of shape [patch_size, patch_size, channels]
    """

    image = neighbor_time_series[:, :, index_o * n_channels: (index_o + 1) * n_channels]
    image.set_shape([patch_size, patch_size, n_channels])

    return image


def extract_neighborhood(input_tensor, img_size, patch_size, n_channels_frames, neighborhood_factor=2):
    """
    This method extracts a neighborhood of shape [2*patch_size, 2*patch_sizes]

    :param input_tensor: A tensor of shape [img_size, img_size, n_channels_frames]
    :param img_size: Height and width of the original image
    :param patch_size: Height and width of the required patch
    :param n_channels_frames: Number of channels*frames in the tensor
    :param neighborhood_factor: Neighborhood size

    :return: A tensor of shape [patch_size, patch_size, n_channels_frames]
    """

    # Define the neighborhood size
    neighborhood_size = neighborhood_factor * patch_size

    # Get random integers for spatial index
    index = tf.random_uniform(shape=[2], minval=0, maxval=img_size - neighborhood_size, dtype=tf.int32)
    neighborhood = input_tensor[index[0]:index[0] + neighborhood_size, index[1]:index[1] + neighborhood_size, :]
    neighborhood.set_shape([neighborhood_size, neighborhood_size, n_channels_frames])

    return neighborhood, index


def get_random_index(neighborhood_factor=2):
    """
    This method generates a random index tensor [x, y] whose values are between 0 and neighborhood_factor
    """
    index = tf.random_uniform(shape=[2], minval=0, maxval=neighborhood_factor, dtype=tf.int32)
    return index


def extract_time_series(neighborhood, patch_size, n_channels_frames, neighborhood_factor=2):
    """
    This method extracts two time series from a neighborhood

    :param neighborhood: A tensor of shape [neighborhood_size, neighborhood_size, n_channels_frames]
    :param patch_size: Height and width of the required patch
    :param n_channels_frames: Number of channels*frames in the tensor
    :param neighborhood_factor: Neighborhood size

    :return: A tensor of shape [patch_size, patch_size, n_channels_frames]
    """

    # Get index to define the time series to take from the neighborhood
    cond = lambda x, y: tf.reduce_all(tf.equal(x, y))
    body = lambda x, y: (x, get_random_index())

    index0 = get_random_index(neighborhood_factor)
    index1 = get_random_index(neighborhood_factor)

    new_index0, new_index1 = tf.while_loop(cond=cond, body=body, loop_vars=(index0, index1))

    time_series = neighborhood[new_index0[0] * patch_size:(new_index0[0] + 1) * patch_size,
                  new_index0[1] * patch_size:(new_index0[1] + 1) * patch_size, :]
    time_series.set_shape([patch_size, patch_size, n_channels_frames])

    neighbor_time_series = neighborhood[new_index1[0] * patch_size:(new_index1[0] + 1) * patch_size,
                           new_index1[1] * patch_size:(new_index1[1] + 1) * patch_size, :]
    neighbor_time_series.set_shape([patch_size, patch_size, n_channels_frames])

    return time_series, neighbor_time_series, new_index0, new_index1


def extract_tuples(inputs, n_patches, img_size, patch_size, n_channels, neighborhood_factor=2):
    """
    This method extracts randomly <n_patches> patches from inputs.

    :param inputs: Input tensor used as data source. Input shape [batch_size, img_size, img_size, n_frames * channels]
    :param n_patches: Number of patches to be extracted
    :param img_size: Height and width of the image
    :param patch_size: Height and width of the required patch
    :param n_channels: Number of channels in each image from the time series

    :return: Three tensors of shape [n_patches, patch_size, patch_size, channels]
    """

    # Get the number of patches per input
    batch_size, height, width, channels_frames = inputs.get_shape().as_list()
    n_frames = channels_frames // n_channels
    n_samples = n_patches // batch_size

    if n_patches % batch_size != 0:
        raise ValueError('Check the batch_size and the n_patches parameter values')

    # For each input get n_samples of size patch_size x patch_size
    reference_list = []
    same_location_list = []
    same_time_list = []

    for i in range(batch_size):
        data = inputs[i]
        for j in range(n_samples):
            neighborhood, _ = extract_neighborhood(data, img_size, patch_size, channels_frames, neighborhood_factor)
            time_series, neighbor_time_series, _, _ = extract_time_series(neighborhood, patch_size, channels_frames, neighborhood_factor)
            image_o, image_f, index_o, index_f = extract_same_location_images(time_series, n_channels, n_frames, patch_size)
            image_t = extract_same_time_image(neighbor_time_series, n_channels, patch_size, index_o)

            reference_list.append(image_o)
            same_location_list.append(image_f)
            same_time_list.append(image_t)

    reference_batch = tf.stack(reference_list)
    same_location_batch = tf.stack(same_location_list)
    same_time_batch = tf.stack(same_time_list)

    return reference_batch, same_location_batch, same_time_batch
