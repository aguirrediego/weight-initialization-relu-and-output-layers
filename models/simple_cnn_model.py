from __future__ import print_function
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.models import Model


class SipleCNNModel:

    @staticmethod
    def get_model(instance_shape, num_classes):
        input_layer = Input(shape=instance_shape)
        x = Conv2D(32, (3, 3), padding='same', activation='relu')(input_layer)
        x = Conv2D(32, (3, 3), activation='relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.25)(x)
        x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
        x = Conv2D(64, (3, 3), activation='relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.5)(x)
        x = Flatten()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(num_classes, activation='softmax')(x)

        model = Model(input_layer, x)

        return model
