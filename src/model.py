import GeoSimilarity.src.nets as nets
from tensorflow.contrib.gan.python.eval.python import eval_utils
from tensorflow.python.summary import summary


class Model(object):
    """
    Create a model for training/testing
    """
    def __init__(self, model_type, is_train, grid_size):
        self.model_type = model_type
        self.is_train = is_train
        self.grid_size = grid_size

    def __call__(self, img_a, img_b, labels):
        # Save inputs as attributes
        self.img_a = img_a
        self.img_b = img_b
        self.labels = labels

        # Choose a model type
        if self.model_type == 'early_fusion':
            early_fusion = nets.EarlyFusion(name='early_fusion', is_train=self.is_train)
            self.end_points, self.outputs = early_fusion(inputs_a=self.img_a, inputs_b=self.img_b)
            self.var_list = early_fusion.var_list

        else:  # self.model_type == 'late_fusion'
            late_fusion = nets.LateFusion(name='late_fusion', is_train=self.is_train)
            self.end_points, self.outputs = late_fusion(inputs_a=self.img_a, inputs_b=self.img_b)
            self.var_list = late_fusion.var_list

    def build(self, img_a, img_b, labels):
        self.__call__(img_a, img_b, labels)

        # Add summaries
        num_images = self.grid_size ** 2
        image_shape = self.img_a.get_shape().as_list()[1:3]

        summary.image('image_a',
                      eval_utils.image_grid(self.img_a[:num_images, :, :, :3],
                                            grid_shape=(self.grid_size, self.grid_size),
                                            image_shape=image_shape,
                                            num_channels=3),
                      max_outputs=1)

        summary.image('image_b',
                      eval_utils.image_grid(self.img_b[:num_images, :, :, :3],
                                            grid_shape=(self.grid_size, self.grid_size),
                                            image_shape=image_shape,
                                            num_channels=3),
                      max_outputs=1)