from __future__ import print_function
import keras


class VGG19Model:

    @staticmethod
    def get_model(instance_shape, num_classes):
        return keras.applications.vgg19.VGG19(include_top=True, weights=None,
                                input_tensor=None, input_shape=instance_shape, pooling=None, classes=num_classes)


