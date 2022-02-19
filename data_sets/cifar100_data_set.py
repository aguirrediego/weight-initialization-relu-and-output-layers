import keras
from keras.datasets import cifar100
import random
import tensorflow as tf

class CIFAR100:
    num_classes = 100
    image_width = 32
    image_height = 32
    image_channels = 3
    num_training_examples = 50000
    num_testing_examples = 10000

    @staticmethod
    def get_data_set():

        (x_train, y_train), (x_test, y_test) = cifar100.load_data()
        y_train = tf.keras.utils.to_categorical(y_train, CIFAR100.num_classes)
        y_test = tf.keras.utils.to_categorical(y_test, CIFAR100.num_classes)

        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255

        return x_train, y_train, x_test, y_test

    @staticmethod
    def get_input_shape():
        return CIFAR100.image_width, CIFAR100.image_height, CIFAR100.image_channels

    @staticmethod
    def get_random_training_subset(subset_size, random_seed):

        if random_seed is not None:
            random.seed(random_seed)

        (x_train, y_train), (_, _) = cifar100.load_data()
        y_train = tf.keras.utils.np_utils.to_categorical(y_train, CIFAR100.num_classes)

        x_train = x_train.astype('float32')
        x_train /= 255

        return CIFAR100.get_random_subset(subset_size, random_seed, x_train, y_train)

    @staticmethod
    def get_random_testing_subset(subset_size, random_seed):

        if random_seed is not None:
            random.seed(random_seed)

        (_, _), (x_test, y_test) = cifar100.load_data()
        y_test = tf.keras.utils.to_categorical(y_test, CIFAR100.num_classes)

        x_test = x_test.astype('float32')
        x_test /= 255

        return CIFAR100.get_random_subset(subset_size, random_seed, x_test, y_test)

    @staticmethod
    def get_random_subset(subset_size, random_seed, x_set, y_set):

        if random_seed is not None:
            random.seed(random_seed)

        indexes = list(range(x_set.shape[0]))

        random_indexes = random.sample(indexes, subset_size)

        x_subset = x_set[random_indexes]
        y_subset = y_set[random_indexes]

        return x_subset, y_subset
