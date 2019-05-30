from __future__ import print_function
import keras


class InceptionV3:

    @staticmethod
    def get_model(instance_shape, num_classes):
        return keras.applications.inception_v3.InceptionV3(include_top=True, weights=None,
                                input_tensor=None, input_shape=instance_shape, pooling=None, classes=num_classes)


