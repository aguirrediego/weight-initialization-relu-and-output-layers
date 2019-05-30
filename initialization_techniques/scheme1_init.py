from __future__ import print_function
import numpy as np
import math
from utils.utils import Utils
from keras.layers import Dense, Convolution2D, Convolution1D, Convolution3D, Conv1D, Conv2D, Conv3D


class WeightInitScheme1Params:
    def __init__(self, batch, use_gram_schmidt, verbose, active_frac, goal_std):
        self.batch = batch
        self.use_gram_schmidt = use_gram_schmidt
        self.verbose = verbose
        self.active_frac = active_frac
        self.goal_std = goal_std


class WeightInitScheme1:

    @staticmethod
    def initialize(model, params):

        batch = params.batch
        active_frac = params.active_frac
        goal_std = params.goal_std
        use_gram_schmidt = params.use_gram_schmidt if params.use_gram_schmidt is not None else False
        verbose = params.verbose if params.verbose is not None else True

        layers_initialized = 0

        if verbose:
            print("------- Scheme 1 - Initialization Process Started ------- ")

        for i in range(len(model.layers)):
            layer = model.layers[i]

            try:
                classes_to_consider = (Dense, Convolution2D, Convolution1D, Convolution3D, Conv1D, Conv2D, Conv3D)

                if not isinstance(layer, classes_to_consider):
                    if verbose:
                        print("Scheme1 - skipping " + layer.name + ' - not in the list of classes to be initialized')
                    continue

                weights_and_biases = layer.get_weights()

                last_dim = weights_and_biases[0].shape[-1]

                # Step 1 - Orthonormalization
                # Forcing weight tensor to be a 2D matrix, so we can make each column orthogonal to each other
                weights_reshaped = weights_and_biases[0].reshape((-1, last_dim))
                if use_gram_schmidt:
                    weights_reshaped = Utils.gram_schmidt(weights_reshaped, False, False)
                else:
                    weights_reshaped = Utils.svd_orthonormal(weights_reshaped.shape)

                weights_and_biases[0] = np.reshape(weights_reshaped, weights_and_biases[0].shape)
                weights_and_biases[1] = np.zeros(weights_and_biases[1].shape)

                layer.set_weights(weights_and_biases)

                # Step 2 - ReLU Adaptation
                if active_frac is not None:
                    # Get layer's activations before ReLU
                    raw = Utils.get_layer_linear_activations(model, layer, batch)
                    raw = raw.reshape((-1, raw.shape[-1]))

                    # Sort all columns in the activation matrix
                    r = raw.shape[0]
                    sorted_raw = np.sort(raw, axis=0)

                    # The bias for each unit is set to the negative of the nth value in the activations for that unit,
                    # where n is given by the active_frac hyper-parameter
                    new_biases = -sorted_raw[math.floor(r - active_frac * r), :]

                    weights_and_biases[1] = new_biases
                    layer.set_weights(weights_and_biases)

                # Step 3 - Standarization
                if goal_std is not None:
                    # Get layer's activations using the initialization set
                    activations = Utils.get_layer_activations(model, layer, batch)
                    activations = activations.reshape((-1, activations.shape[-1]))

                    h1_s_std = np.std(activations, axis=0)

                    weights_and_biases = layer.get_weights()
                    last_dim = weights_and_biases[0].shape[-1]
                    new_weights = weights_and_biases[0].reshape((-1, last_dim))
                    new_biases = weights_and_biases[1]

                    # Compute new weights/biases - dive them by std of the activation and multiplying times desired std
                    for j in range(new_weights.shape[1]):
                        new_weights[:, j] = new_weights[:, j] / h1_s_std[j] * goal_std
                        new_biases[j] = new_biases[j] / h1_s_std[j] * goal_std

                    new_weights = np.reshape(new_weights, weights_and_biases[0].shape)
                    weights_and_biases[0] = new_weights
                    weights_and_biases[1] = new_biases

                    layer.set_weights(weights_and_biases)

                # Print some statistics about the weights/biases and the layer's activations
                if verbose:
                    weights_and_biases = layer.get_weights()

                    new_weights = weights_and_biases[0].reshape((-1, last_dim))
                    new_biases = weights_and_biases[1]

                    activations = Utils.get_layer_activations(model, layer, batch)
                    activations = activations.reshape((-1))

                    new_weights = new_weights.reshape((-1, new_weights.shape[-1]))
                    new_biases = new_biases.reshape((-1, new_biases.shape[-1]))

                    print("------- Scheme 1 - Layer initialized: " + layer.name + " ------- ")

                    print("Weights -- Std: ", np.std(new_weights), " Mean: ", np.mean(new_weights), " Max: ",
                          np.max(new_weights), " Min: ", np.min(new_weights))

                    print("Biases -- Std: ", np.std(new_biases), " Mean: ", np.mean(new_biases), " Max: ",
                          np.max(new_biases), " Min: ", np.min(new_biases))

                    print("Layer activations' std: ", np.std(activations, axis=0))
                    print("Layer activations <= 0: ", (len(activations[activations <= 0]) / len(activations)))
                    print("Layer activations >  0: ", (len(activations[activations > 0]) / len(activations)))

                layers_initialized += 1
            except Exception as ex:

                print("Could not initialize layer: ", layer.name, " Error: ", ex)
                continue

        if verbose:
            print("------- Scheme 1 - DONE - total layers initialized ", layers_initialized, "------- ")

        return model
