from sklearn.cluster import KMeans
import numpy as np
import keras
import os
from keras.models import load_model
from keras.models import Model
from sklearn.neighbors import NearestNeighbors


class Utils:

    @staticmethod
    def create_init_set_kmeans(x_input_set, y_input_set, num_elements_in_init_set, random_seed):
        # Assumption: First dimension in input_set_x is the number of instances
        # Assumption: One-hot encoding used in input_set_y

        encoding_set = np.unique(y_input_set, axis=0)

        x_desired_shape = list(x_input_set.shape)
        x_desired_shape[0] = 0

        y_desired_shape = list(y_input_set.shape)
        y_desired_shape[0] = 0

        x_init_set = np.empty(x_desired_shape)
        y_init_set = np.empty(y_desired_shape)

        for i in range (encoding_set.shape[0]):
            instance_indexes = np.where((y_input_set == encoding_set[i]).all(axis=1))[0]
            x_sub_set = x_input_set[instance_indexes]

            fraction_instances = x_sub_set.shape[0] / x_input_set.shape[0]
            num_clusters = int(round(num_elements_in_init_set * fraction_instances))

            x_sub_set_reshaped = x_sub_set.reshape(x_sub_set.shape[0], -1)

            kmeans = KMeans(n_clusters=num_clusters, random_state=random_seed).fit(x_sub_set_reshaped)
            neigh = NearestNeighbors(n_neighbors=1)
            neigh.fit(x_sub_set_reshaped)

            centers = kmeans.cluster_centers_

            init_set_neighs_indexes = neigh.kneighbors(centers, n_neighbors=1, return_distance=False)
            init_set_neighs = x_sub_set_reshaped[init_set_neighs_indexes]

            desired_shape = list(x_sub_set.shape)
            desired_shape[0] = num_clusters

            x_init_set_encoding_i = init_set_neighs.reshape(desired_shape)
            y_init_set_encoding_i = np.tile(encoding_set[i], (num_clusters, 1))

            x_init_set = np.concatenate((x_init_set, x_init_set_encoding_i), axis=0)
            y_init_set = np.concatenate((y_init_set, y_init_set_encoding_i), axis=0)

        return x_init_set, y_init_set

    @staticmethod
    def gram_schmidt(X, row_vecs=True, norm=True):
        if not row_vecs:
            X = X.T
        Y = X[0:1, :].copy()
        for i in range(1, X.shape[0]):
            proj = np.diag((X[i, :].dot(Y.T) / np.linalg.norm(Y, axis=1) ** 2).flat).dot(Y)
            Y = np.vstack((Y, X[i, :] - proj.sum(0)))
        if norm:
            Y = np.diag(1 / np.linalg.norm(Y, axis=1)).dot(Y)
        if row_vecs:
            return Y
        else:
            return Y.T

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
    def get_layer_linear_activations(model, layer, x_batch):

        model_path = os.path.join('temp.h5')
        try:
            model.save(model_path)
            inter_model = load_model(model_path)

            inter_layer = inter_model.get_layer(layer.name)
            inter_layer.activation = keras.activations.linear

            inter_model.save(model_path)
            inter_model = load_model(model_path)

            activations = Utils.get_layer_activations(inter_model, inter_model.get_layer(layer.name),
                                                      x_batch)
            return activations
        finally:
            os.remove(model_path)

    @staticmethod
    def get_layer_activations(model, layer, x_batch):
        intermediate_layer_model = Model(
            inputs=model.get_input_at(0),
            outputs=layer.get_output_at(0)
        )
        activations = intermediate_layer_model.predict(x_batch)
        del intermediate_layer_model
        return activations

    @staticmethod
    def get_tensor_activations(model, outputs, x_batch):

        if model.get_input_at(0) == outputs:
            return x_batch

        intermediate_layer_model = Model(
            inputs=model.get_input_at(0),
            outputs=outputs
        )

        activations = intermediate_layer_model.predict(x_batch)
        del intermediate_layer_model
        return activations


def test_utils():
    li = [
        [0, 0, 1],
        [0, 0, 1],
        [0, 0, 1],


        [1, 0, 0],
        [1, 0, 0],
        [1, 0, 0],


        [0, 1, 0],
        [0, 1, 0],
        [0, 1, 0]

    ]
    y = np.asarray(li)

    x = np.random.rand(y.shape[0], 2, 4, 5)

    init_set_x, init_set_y = Utils.create_init_set_kmeans(x, y, 3, 7)

    print("init set")
    print(init_set_x)
    print(init_set_y)

    print("origian set")
    print(x)


if __name__ == "__main__":
    test_utils()

