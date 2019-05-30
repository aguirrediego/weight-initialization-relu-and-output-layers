from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dropout, Activation, Flatten
from keras.layers import MaxPooling2D
from keras.datasets import cifar10
from initialization_techniques.scheme4_init import *
from initialization_techniques.scheme1_init import *

import keras
import random


def main():

    # Read the data set
    num_classes = 10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    # Create the model
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',
                     input_shape=x_train[0].shape))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

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

    # Initiate Stochastic Gradient Descent optimizer

    opt = keras.optimizers.sgd(lr=0.01)

    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    loss_and_acc = model.test_on_batch(x_test, y_test)

    initial_val_loss = loss_and_acc[0]
    initial_val_acc = loss_and_acc[1]

    train_history = model.fit(x_train, y_train,
                              batch_size=128,
                              epochs=1,
                              verbose=1,
                              validation_data=(x_test, y_test),
                              shuffle=True)

    val_loss = train_history.history['val_loss']
    val_acc = train_history.history['val_acc']

    val_loss.insert(0, initial_val_loss)
    val_acc.insert(0, initial_val_acc)

    print("Loss: ", val_loss)
    print("Accuracy: ", val_acc)


if __name__ == '__main__':
    main()
