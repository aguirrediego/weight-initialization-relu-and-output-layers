from __future__ import print_function
from keras import backend as keras_backend
import numpy as np
from tensorflow.python.ops import random_ops
import math
from keras.layers import Dense, Convolution2D, Convolution1D, Convolution3D, Conv1D, Conv2D, Conv3D


class WeightInitHeParams:
    def __init__(self, verbose, random_seed, init_hidden):
        self.verbose = verbose
        self.random_seed = random_seed
        self.init_hidden = init_hidden


class WeightInitHe:

    @staticmethod
    def compute_fans(shape):
        """Computes the number of input and output units for a weight shape.
        Args:
          shape: Integer shape tuple or TF tensor shape.
        Returns:
          A tuple of scalars (fan_in, fan_out).
        """
        if len(shape) < 1:  # Just to avoid errors for constants.
            fan_in = fan_out = 1
        elif len(shape) == 1:
            fan_in = fan_out = shape[0]
        elif len(shape) == 2:
            fan_in = shape[0]
            fan_out = shape[1]
        else:
            # Assuming convolution kernels (2D, 3D, or more).
            # kernel shape: (..., input_depth, depth)
            receptive_field_size = 1.
            for dim in shape[:-2]:
                receptive_field_size *= dim
            fan_in = shape[-2] * receptive_field_size
            fan_out = shape[-1] * receptive_field_size

        return fan_in, fan_out

    @staticmethod
    def sample_from_truncated_normal(mode, shape, scale, seed):
        fan_in, fan_out = WeightInitHe.compute_fans(shape)
        if mode == "fan_in":
            scale /= max(1., fan_in)
        elif mode == "fan_out":
            scale /= max(1., fan_out)
        else:
            scale /= max(1., (fan_in + fan_out) / 2.)

        # constant taken from scipy.stats.truncnorm.std(a=-2, b=2, loc=0., scale=1.)
        stddev = math.sqrt(scale) / .87962566103423978

        return random_ops.truncated_normal(
            shape=shape, mean=0.0, stddev=stddev, seed=seed).eval(session=keras_backend.get_session())

    @staticmethod
    def initialize(model, params):
        random_seed = params.random_seed
        verbose = params.verbose
        init_hidden = params.init_hidden

        layers_initialized = 0

        if verbose:
            print("------- He - Initialization Process Started ------- ")

        for i in range(len(model.layers)):
            layer = model.layers[i]

            layer_index = i

            if not init_hidden:  # If initializing output layer(s) only
                if not hasattr(layer, 'activation') or "softmax" not in str(layer.activation):
                    continue

                while not isinstance(layer, Dense) and layer_index > 0:
                    layer_index -= 1
                    layer = model.layers[layer_index]

            try:

                classes_to_consider = (Dense, Convolution2D, Convolution1D, Convolution3D, Conv1D, Conv2D, Conv3D)
                if not isinstance(layer, classes_to_consider):
                    if verbose:
                        print("HE - skipping " + layer.name + ' - not in the list of classes to be initialized')
                    continue
                weights_and_biases = layer.get_weights()

                for weight_index in range(len(weights_and_biases)):
                    weight_tensor = weights_and_biases[weight_index]
                    weights_and_biases[weight_index] = WeightInitHe.sample_from_truncated_normal(
                        mode="fan_in", shape=np.array(weight_tensor.shape), scale=2., seed=random_seed)

                    random_seed = None

                    w = weights_and_biases[weight_index]
                    w_std = np.std(w)
                    w_mean = np.mean(w)

                    if verbose:
                        print("Weights -- Std: ", w_std, " Mean: ", w_mean, " Max: ",
                              np.max(w), " Min: ", np.min(w))

                layer.set_weights(weights_and_biases)

                layers_initialized += 1

                if verbose:
                    print("------- He - Layer initialized: ", layer.name, " ------- ")

            except Exception as ex:
                print("Exception thrown: ", ex)
                continue

        if verbose:
            print("------- He - DONE - total layers initialized: ", layers_initialized, " ------- ")

        return model
