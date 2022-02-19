from __future__ import print_function
import tensorflow as tf
from tensorflow.keras.layers import Dropout, Activation, Flatten, MaxPooling2D, Input, Conv2D, Dense
from tensorflow.keras.models import Model

from tensorflow.keras.datasets import cifar10
from keras.utils import np_utils
from initialization_techniques.scheme4_init import *
from initialization_techniques.scheme1_init import *

import random


def main():

    # Read the data set
    num_classes = 10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    x_train = x_train / 255.0
    x_test = x_test / 255.0

    y_train = y_train.flatten()
    y_test = y_test.flatten()

    # Create the model
    input_layer = Input(shape=x_train[0].shape)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    x = Conv2D(32, (3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.25)(x)
    x = Dense(num_classes)(x)

    model = Model(input_layer, x)

    # Initiate Stochastic Gradient Descent optimizer
    opt = tf.keras.optimizers.SGD(learning_rate=0.001)

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    # Initialization Logic
    initialization_set_size = 2048
    indexes = list(range(x_train.shape[0]))

    random_indexes = random.sample(indexes, initialization_set_size)

    init_set_x = x_train[random_indexes]
    init_set_y = y_train[random_indexes]

    # - Defining the parameters of the initialization technique for ReLU layers
    relu_layers_init_params = WeightInitScheme1Params(
        batch=init_set_x,
        use_gram_schmidt=False,
        verbose=True,
        active_frac=0.8,
        goal_std=1.0
    )

    # - Run initialization process
    model = WeightInitScheme1.initialize(model, relu_layers_init_params)

    # Defining the parameters of the initialization technique for the output layer
    output_layer_init_params = WeightInitScheme4Params(
        batch_x=init_set_x,
        batch_y=init_set_y,
        verbose=True
    )

    # - Run initialization process
    model = WeightInitScheme4.initialize(model, output_layer_init_params)

    # Compute loss and accuracy on initialization dataset
    loss_and_acc = model.test_on_batch(init_set_x, init_set_y)
    init_set_loss, init_set_acc = loss_and_acc[0], loss_and_acc[1]
    print("init_set_loss:", init_set_loss, "init_set_acc", init_set_acc)

    loss_and_acc = model.test_on_batch(x_test, y_test)
    initial_val_loss, initial_val_acc = loss_and_acc[0], loss_and_acc[1]
    print("initial_val_loss:", initial_val_loss, "initial_val_acc", initial_val_acc)

    train_history = model.fit(x_train, y_train,
                              batch_size=128,
                              epochs=10,
                              verbose=1,
                              validation_data=(x_test, y_test),
                              shuffle=True)

    val_loss = train_history.history['val_loss']
    val_acc = train_history.history['val_accuracy']

    val_loss.insert(0, initial_val_loss)
    val_acc.insert(0, initial_val_acc)

    print("Loss: ", val_loss)
    print("Accuracy: ", val_acc)


if __name__ == '__main__':
    main()
