from enum import Enum
from data_sets.cifar10_data_set import CIFAR10
from data_sets.cifar100_data_set import CIFAR100
from data_sets.imagenet_data_set import ImageNet
from data_sets.leap_gest_recog_data_set import LeapGestRecog

class DataSetOptions(Enum):
    CIFAR10 = 1
    CIFAR100 = 2
    ImageNet = 3
    LeapGestRecog = 4


class DataSetManager:

    data_set_dictionary = {
        DataSetOptions.CIFAR10: CIFAR10,
        DataSetOptions.CIFAR100: CIFAR100,
        DataSetOptions.ImageNet: ImageNet,
        DataSetOptions.LeapGestRecog : LeapGestRecog

    }

    @staticmethod
    def get_data_set(data_set_option):

        data_set_class = DataSetManager.data_set_dictionary.get(data_set_option, CIFAR10)
        x_train, y_train, x_test, y_test = data_set_class.get_data_set()

        return x_train, y_train, x_test, y_test

    @staticmethod
    def get_random_training_subset(subset_size, random_seed, data_set_option):

        data_set_class = DataSetManager.data_set_dictionary.get(data_set_option, CIFAR10)
        return data_set_class.get_random_training_subset(subset_size, random_seed)

    @staticmethod
    def get_random_testing_subset(subset_size, random_seed, data_set_option):

        data_set_class = DataSetManager.data_set_dictionary.get(data_set_option, CIFAR10)
        return data_set_class.get_random_testing_subset(subset_size, random_seed)

    @staticmethod
    def get_num_classes(data_set_option):

        data_set_class = DataSetManager.data_set_dictionary.get(data_set_option, CIFAR10)
        return data_set_class.num_classes

    @staticmethod
    def get_input_shape(data_set_option):

        data_set_class = DataSetManager.data_set_dictionary.get(data_set_option, CIFAR10)
        return data_set_class.get_input_shape()

    @staticmethod
    def get_num_training_examples(data_set_option):

        data_set_class = DataSetManager.data_set_dictionary.get(data_set_option, CIFAR10)
        return data_set_class.num_training_examples

    @staticmethod
    def get_num_testing_examples(data_set_option):
        data_set_class = DataSetManager.data_set_dictionary.get(data_set_option, CIFAR10)
        return data_set_class.num_testing_examples

def test_data_set_manager():
    x_train, y_train, x_test, y_test = DataSetManager.get_data_set(DataSetOptions.CIFAR10)

    print(x_train, y_train, x_test, y_test )


# test_data_set_manager()
