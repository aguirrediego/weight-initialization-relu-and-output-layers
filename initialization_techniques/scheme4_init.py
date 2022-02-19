from __future__ import print_function
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from utils.utils import Utils
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
import multiprocessing


class WeightInitScheme4Params:
    def __init__(self, batch_x, batch_y, verbose=True, use_linear_svm=False):
        self.batch_x = batch_x
        self.batch_y = batch_y
        self.verbose = verbose
        self.use_linear_svm = use_linear_svm


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
        verbose = params.verbose
        use_linear_svm = params.use_linear_svm

        layers_initialized = 0

        if verbose:
            print("------- Scheme 4 - Initialization Process Started ------- ")

        # Assumption: Last dense layer is the output layer
        layer = model.layers[-1]
        prev_layer = model.layers[-2]

        j = len(model.layers) - 2
        while not isinstance(layer, Dense) and j >= 0:
            layer = model.layers[j]
            prev_layer = model.layers[j - 1]
            j -= 1

        try:
            weights_and_biases = layer.get_weights()

            activations = Utils.get_layer_activations(model, prev_layer, batch_x)

            if use_linear_svm:
                clf = OneVsRestClassifier(LinearSVC(), n_jobs=-1)

                if batch_y.ndim > 1:
                    batch_y = tf.argmax(batch_y, axis=1)

                clf.fit(activations, batch_y)

                new_weights, new_biases = np.transpose(clf.coef_), clf.intercept_.reshape(-1)
            else:
                if batch_y.ndim == 1:
                    batch_y = tf.keras.utils.to_categorical(batch_y)

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

                print("Layer activations' std: ", np.std(activations, axis=0))
                print("Layer activations <= 0: ", (len(activations[activations <= 0]) / len(activations)))
                print("Layer activations >  0: ", (len(activations[activations > 0]) / len(activations)))

        except Exception as ex:
            print("Could not initialize layer: " + layer.name)
            print(ex)

        if verbose:
            print("------- Scheme 4 - DONE - total layers initialized: ", layers_initialized, " ------- ")

        return model
