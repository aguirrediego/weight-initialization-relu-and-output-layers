import keras
from keras.preprocessing.image import img_to_array, load_img
import os
import random
import numpy


class ImageNet:

    data_set_path = "~/PhD/imagenet_256"
    num_classes = 1000
    image_width = 256
    image_height = 256
    image_channels = 3

    training_images = []
    training_labels = []

    testing_images = []
    testing_labels = []

    data_set_initialized = False
    num_training_examples = 1281167
    num_testing_examples = 5000

    @staticmethod
    def get_input_shape():
        return ImageNet.image_width, ImageNet.image_height, ImageNet.image_channels

    @staticmethod
    def get_dirs_and_files(path):
        dir_list = [directory for directory in os.listdir(path) if os.path.isdir(path + '/' + directory)]
        file_list = [directory for directory in os.listdir(path) if not os.path.isdir(path + '/' + directory)]

        return dir_list, file_list

    @staticmethod
    def read_data_set_image_paths(path):
        class_counter = 0

        x_set = []
        y_set = []

        dir_list, file_list = ImageNet.get_dirs_and_files(path)
        dir_list.sort()
        for folder in dir_list:

            folder_path = os.path.join(path, folder)
            sub_dir_list, sub_file_list = ImageNet.get_dirs_and_files(folder_path)

            for file in sub_file_list:
                x_set.append(os.path.join(folder_path, file))
                y_set.append(class_counter)

            class_counter = class_counter + 1

        return x_set, y_set

    @staticmethod
    def init_data_set_image_paths():
        train_path = os.path.join(ImageNet.data_set_path, "train")
        test_path = os.path.join(ImageNet.data_set_path, "test")

        ImageNet.training_images, ImageNet.training_labels = ImageNet.read_data_set_image_paths(train_path)
        ImageNet.testing_images, ImageNet.testing_labels = ImageNet.read_data_set_image_paths(test_path)

        ImageNet.data_set_initialized = True

    @staticmethod
    def get_data_set():
        x_train = None
        y_train = None
        x_test = None
        y_test = None

        return x_train, y_train, x_test, y_test

    @staticmethod
    def get_random_training_subset(subset_size, random_seed):
        if not ImageNet.data_set_initialized:
            ImageNet.init_data_set_image_paths()

        return ImageNet.get_random_subset(subset_size, random_seed, ImageNet.training_images, ImageNet.training_labels)

    @staticmethod
    def get_random_testing_subset(subset_size, random_seed):

        if not ImageNet.data_set_initialized:
            ImageNet.init_data_set_image_paths()

        return ImageNet.get_random_subset(subset_size, random_seed, ImageNet.testing_images, ImageNet.testing_labels)

    @staticmethod
    def get_random_subset(subset_size, random_seed, x_set, y_set):
        if random_seed is not None:
            random.seed(random_seed)

        indexes = list(range(len(x_set)))

        random_indexes = random.sample(indexes, subset_size)

        x_subset = numpy.zeros(shape=(subset_size, ImageNet.image_width,
                                      ImageNet.image_height, ImageNet.image_channels))

        y_subset = numpy.zeros(shape=subset_size)

        subset_index = 0
        for random_index in random_indexes:
            try:
                img = load_img(x_set[random_index])
                x_subset[subset_index] = img_to_array(img)
                x_subset[subset_index] /= 255

                y_subset[subset_index] = y_set[random_index]

                subset_index += 1

            except Exception as ex:
                print("Exception thrown while reading an ImageNet sample: ", ex)
                print("Ignore image {}".format(x_set[random_index]))

        return x_subset, keras.utils.to_categorical(y_subset, ImageNet.num_classes)
