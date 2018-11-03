import tensorflow as tf
import GeoSimilarity.src.nets as nets
import GeoSimilarity.src.inputs as inputs


class TestNets(tf.test.TestCase):

    def test_buid_fnet(self):
        print('\ntest_buid_fnet')

        batch_size = 64
        img_size = 64
        channels = 4

        dataset = inputs.Input()
        img_a, img_b = dataset.get_batch(batch_size=batch_size, img_size=img_size, channels=channels)

        fnet = nets.FeatureNet(name='feature_net', is_train=True)
        end_points_a, outputs_a = fnet(inputs=img_a)
        end_points_b, outputs_b = fnet(inputs=img_b)

        print('\nEnd points A\n')
        for end_point in end_points_a:
            print('##############')
            print(end_point)
            print(end_points_a[end_point].get_shape().as_list())

        print('\nEnd points B\n')
        for end_point in end_points_b:
            print('##############')
            print(end_point)
            print(end_points_b[end_point].get_shape().as_list())

        features = nets.get_fnet_feature_map()
        assert outputs_a.get_shape().as_list() == [batch_size, img_size / 2 ** len(features), img_size / 2 ** len(features), features[-1]]
        assert outputs_b.get_shape().as_list() == [batch_size, img_size / 2 ** len(features), img_size / 2 ** len(features), features[-1]]

    def test_buid_dnet(self):
        print('\ntest_buid_dnet')

        batch_size = 64
        feature_size = 8
        channels = 512

        dataset = inputs.Input()
        feature_a, feature_b = dataset.get_batch(batch_size=batch_size, img_size=feature_size, channels=channels)

        dnet = nets.DecisionNet(name='decision_net', is_train=True)
        end_points, outputs = dnet(inputs_a=feature_a, inputs_b=feature_b, concat=True)

        print('\nEnd points\n')
        for end_point in end_points:
            print('##############')
            print(end_point)
            print(end_points[end_point].get_shape().as_list())

        assert outputs.get_shape().as_list() == [batch_size, 1]

    def test_buid_earlyfusion(self):
        print('\ntest_buid_earlyfusion')
        batch_size = 64
        img_size = 64
        channels = 4

        dataset = inputs.Input()
        img_a, img_b = dataset.get_batch(batch_size=batch_size, img_size=img_size, channels=channels)

        early_fusion = nets.EarlyFusion(name='early_fusion', is_train=True)
        end_points, outputs = early_fusion(inputs_a=img_a, inputs_b=img_b)

        print('\nEnd points\n')
        for end_point in end_points:
            print('##############')
            print(end_point)
            print(end_points[end_point].get_shape().as_list())

        assert outputs.get_shape().as_list() == [batch_size, 1]

    def test_buid_latefusion(self):
        print('\ntest_buid_latefusion')
        batch_size = 64
        img_size = 64
        channels = 4

        dataset = inputs.Input()
        img_a, img_b = dataset.get_batch(batch_size=batch_size, img_size=img_size, channels=channels)

        late_fusion = nets.LateFusion(name='late_fusion', is_train=True)
        end_points, outputs = late_fusion(inputs_a=img_a, inputs_b=img_b)

        print('\nEnd points\n')
        for end_point in end_points:
            print('##############')
            print(end_point)
            print(end_points[end_point].get_shape().as_list())

        assert outputs.get_shape().as_list() == [batch_size, 1]