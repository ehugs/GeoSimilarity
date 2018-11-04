import GeoSimilarity.src.nets as nets


class Model(object):
    """
    Create a model for training/testing
    """
    def __init__(self, type, is_train):
        self.type = type
        self.is_train = is_train

    def __call__(self, img_a, img_b, labels):
        # Save inputs as attributes
        self.img_a = img_a
        self.img_b = img_b
        self.labels = labels

        # Choose a model type
        if self.type == 'early_fusion':
            early_fusion = nets.EarlyFusion(name='early_fusion', is_train=self.is_train)
            self.end_points, self.outputs = early_fusion(inputs_a=self.img_a, inputs_b=self.img_b)

        else:  # self.type == 'late_fusion'
            late_fusion = nets.LateFusion(name='late_fusion', is_train=self.is_train)
            self.end_points, self.outputs = late_fusion(inputs_a=self.img_a, inputs_b=self.img_b)

    def build(self, img_a, img_b, labels):
        self.__call__(img_a, img_b, labels)
