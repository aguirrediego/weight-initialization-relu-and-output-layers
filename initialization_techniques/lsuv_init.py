from __future__ import print_function
import numpy as np
from keras.models import Model
from keras.layers import Dense, Convolution2D, Convolution1D, Convolution3D, Conv1D, Conv2D, Conv3D
from utils.utils import Utils


class WeightInitLSUVParams:
    def __init__(self, batch, verbose, margin, max_iter, init_hidden):
        self.batch = batch
        self.verbose = verbose
        self.margin = margin
        self.max_iter = max_iter
        self.init_hidden = init_hidden


class WeightInitLSUV:

    @staticmethod
    def svd_orthonormal(shape):
        # Orthonorm init code is taked from Lasagne
        # https://github.com/Lasagne/Lasagne/blob/master/lasagne/init.py
        if len(shape) < 2:
            raise RuntimeError("Only shapes of length 2 or more are supported.")
        flat_shape = (shape[0], np.prod(shape[1:]))
        a = np.random.standard_normal(flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v
        q = q.reshape(shape)
        return q

    @staticmethod
    def get_activations(model, layer, x_batch):
        intermediate_layer_model = Model(
            inputs=model.get_input_at(0),
            outputs=layer.get_output_at(0)
        )
        activations = intermediate_layer_model.predict(x_batch)
        return activations

    @staticmethod
    def initialize(model, params):
        batch = params.batch
        verbose = params.verbose if params.verbose is not None else True
        margin = params.margin if params.margin is not None else 0.1
        max_iter = params.max_iter if params.max_iter is not None else 10
        init_hidden = params.init_hidden

        # only these layer classes considered for LSUV initialization; add more if needed
        classes_to_consider = (Dense, Convolution2D, Convolution1D, Convolution3D, Conv1D, Conv2D, Conv3D)

        needed_variance = 1.0

        layers_intialized = 0
        if verbose:
            print("------- LSUV - Initialization Process Started ------- ")

        for i in range(len(model.layers)):
            layer = model.layers[i]

            layer_index = i

            if not init_hidden:  # If initializing output layer(s) only
                if not hasattr(layer, 'activation') or "softmax" not in str(layer.activation):
                    continue

                while not isinstance(layer, Dense) and layer_index > 0:
                    layer_index -= 1
                    layer = model.layers[layer_index]
            elif hasattr(layer, 'activation') and "softmax" in str(layer.activation):
                continue

            try:

                if not isinstance(layer, classes_to_consider):
                    if verbose:
                        print("LSUV - skipping ", layer.name, ' - not in the list of classes to be initialized')
                    continue

                # avoid small layers where activation variance close to zero, esp. for small batches
                if np.prod(layer.get_output_shape_at(0)[1:]) < 32:
                    if verbose:
                        print("LSUV - skipping ", layer.name, ' - too small')
                    continue
                if verbose:
                    print('LSUV initializing', layer.name)

                weights_and_biases = layer.get_weights()
                weights_and_biases[0] = WeightInitLSUV.svd_orthonormal(weights_and_biases[0].shape)
                layer.set_weights(weights_and_biases)

                if init_hidden or "softmax" not in str(layer.activation):  # Use regular activation for hidden layers
                    activations = WeightInitLSUV.get_activations(model, layer, batch)
                else:  # Linear activation for output layers to prevent softmax layer from negatively affecting results
                    activations = Utils.get_layer_linear_activations(model, layer, batch)

                variance = np.var(activations)
                iteration = 0
                if verbose:
                    print(variance)
                while abs(needed_variance - variance) > margin:
                    if np.abs(np.sqrt(variance)) < 1e-7:
                        # avoid division by zero
                        break

                    weights_and_biases = layer.get_weights()
                    weights_and_biases[0] /= np.sqrt(variance) / np.sqrt(needed_variance)
                    layer.set_weights(weights_and_biases)

                    if init_hidden or "softmax" not in str(
                            layer.activation):  # Use regular activation for hidden layers
                        activations = WeightInitLSUV.get_activations(model, layer, batch)
                    # Linear activation for output layers to prevent softmax layer from negatively affecting results
                    else:
                        activations = Utils.get_layer_linear_activations(model, layer, batch)

                    variance = np.var(activations)

                    iteration += 1
                    if verbose:
                        print(variance)
                    if iteration >= max_iter:
                        break

                layers_intialized += 1
            except Exception as ex:
                print("Exception thrown: ", ex)
                continue
        if verbose:
            print('LSUV: total layers initialized', layers_intialized)
        return model
