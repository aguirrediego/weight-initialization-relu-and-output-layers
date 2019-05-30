from __future__ import print_function
import numpy as np
from keras.layers import Dense
from utils.utils import Utils


class WeightInitScheme4Params:
    def __init__(self, batch_x, batch_y, verbose):
        self.batch_x = batch_x
        self.batch_y = batch_y
        self.verbose = verbose


class WeightInitScheme4:

    @staticmethod
    def adjusted_weight_mat_linear(input_x, output, delta_output=2.0):
        # Expected logit values [-2,2]
        output = output * 2 * delta_output - delta_output

        # Use pseudo-inverse to initialize weights
        input_x = input_x.reshape((-1, input_x.shape[-1]))
        out_mean = np.mean(output, axis=0)
        weights = np.matrix(np.linalg.pinv(input_x)) * np.matrix(output)

        raw = np.matrix(input_x) * np.matrix(weights)

        h1_s_mean = np.mean(raw, axis=0)
        biases = out_mean - h1_s_mean

        return np.matrix(weights), np.array(biases)[0]

    @staticmethod
    def initialize(model, params):

        batch_x = params.batch_x
        batch_y = params.batch_y
        verbose = params.verbose if params.verbose is not None else True

        layers_initialized = 0

        if verbose:
            print("------- Scheme 4 - Initialization Process Started ------- ")

        for i in range(len(model.layers)):
            layer = model.layers[i]

            try:
                # Scheme is only targeting softmax layers right now
                if not hasattr(layer, 'activation') or "softmax" not in str(layer.activation):
                    continue

                j = i
                while not isinstance(layer, Dense) and j >= 0:
                    layer = model.layers[j]
                    j -= 1

                weights_and_biases = layer.get_weights()

                activations = Utils.get_tensor_activations(model, layer.input, batch_x)

                new_weights, new_biases = WeightInitScheme4.adjusted_weight_mat_linear(activations, batch_y)

                weights_and_biases[0] = new_weights
                weights_and_biases[1] = new_biases

                layer.set_weights(weights_and_biases)

                layers_initialized += 1

                if verbose:
                    activations = Utils.get_layer_linear_activations(model, layer, batch_x)
                    activations = activations.reshape((-1))

                    new_weights = new_weights.reshape((-1, new_weights.shape[-1]))
                    new_biases = new_biases.reshape((-1, new_biases.shape[-1]))

                    print("------- Scheme 4 - Layer initialized: " + layer.name + " ------- ")

                    print("Weights -- Std: ", np.std(new_weights), " Mean: ", np.mean(new_weights), " Max: ",
                          np.max(new_weights), " Min: ", np.min(new_weights))

                    print("Biases -- Std: ", np.std(new_biases), " Mean: ", np.mean(new_biases), " Max: ",
                          np.max(new_biases), " Min: ", np.min(new_biases))

                    print("Layer activations' std: ",  np.std(activations, axis=0))
                    print("Layer activations <= 0: ", (len(activations[activations <= 0]) / len(activations)))
                    print("Layer activations >  0: ", (len(activations[activations > 0]) / len(activations)))

            except Exception as ex:
                print("Could not initialize layer: " + layer.name)
                print(ex)
                continue

        if verbose:
            print("------- Scheme 4 - DONE - total layers initialized: ", layers_initialized, " ------- ")

        return model
