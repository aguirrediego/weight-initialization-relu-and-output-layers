import keras
from keras.preprocessing.image import img_to_array, load_img
import os
import random
import numpy


class LeapGestRecog:

    data_set_path = "/home/ori/PhD/leapGestRecog256"
    num_classes = 10
    image_width = 256
    image_height = 256
    image_channels = 3

    training_images = []
    training_labels = []

    testing_images = []
    testing_labels = []

    data_set_initialized = False
    num_training_examples = 2000 * 9
    num_testing_examples = 2000 * 1

    @staticmethod
    def get_input_shape():
        return LeapGestRecog.image_width, LeapGestRecog.image_height, LeapGestRecog.image_channels

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

        dir_list, file_list = LeapGestRecog.get_dirs_and_files(path)
        dir_list.sort()
        for folder in dir_list:

            folder_path = os.path.join(path, folder)
            sub_dir_list, sub_file_list = LeapGestRecog.get_dirs_and_files(folder_path)

            for file in sub_file_list:
                x_set.append(os.path.join(folder_path, file))
                y_set.append(class_counter)

            class_counter += 1

        return x_set, y_set

    @staticmethod
    def init_data_set_image_paths():
        training_folders = ['00', '01', '02', '03', '04', '05', '06', '07', '08']
        testing_folders = ['09']

        for folder in training_folders:
            x_set, y_set = LeapGestRecog.read_data_set_image_paths(os.path.join(LeapGestRecog.data_set_path, folder))
            LeapGestRecog.training_images.extend(x_set)
            LeapGestRecog.training_labels.extend(y_set)

        for folder in testing_folders:
            x_set, y_set = LeapGestRecog.read_data_set_image_paths(os.path.join(LeapGestRecog.data_set_path, folder))
            LeapGestRecog.testing_images.extend(x_set)
            LeapGestRecog.testing_labels.extend(y_set)

        LeapGestRecog.data_set_initialized = True

    @staticmethod
    def get_data_set():
        x_train = None
        y_train = None
        x_test = None
        y_test = None

        return x_train, y_train, x_test, y_test

    @staticmethod
    def get_random_training_subset(subset_size, random_seed):
        if not LeapGestRecog.data_set_initialized:
            LeapGestRecog.init_data_set_image_paths()

        return LeapGestRecog.get_random_subset(subset_size, random_seed, LeapGestRecog.training_images, LeapGestRecog.training_labels)

    @staticmethod
    def get_random_testing_subset(subset_size, random_seed):

        if not LeapGestRecog.data_set_initialized:
            LeapGestRecog.init_data_set_image_paths()

        return LeapGestRecog.get_random_subset(subset_size, random_seed, LeapGestRecog.testing_images, LeapGestRecog.testing_labels)

    @staticmethod
    def get_random_subset(subset_size, random_seed, x_set, y_set):
        if random_seed is not None:
            random.seed(random_seed)

        indexes = list(range(len(x_set)))

        random_indexes = random.sample(indexes, subset_size)

        x_subset = numpy.zeros(shape=(subset_size, LeapGestRecog.image_width, LeapGestRecog.image_height, LeapGestRecog.image_channels))
        y_subset = numpy.zeros(shape=(subset_size))

        subset_index = 0
        for random_index in random_indexes:
            try:
                img = load_img(x_set[random_index])
                x_subset[subset_index] = img_to_array(img)
                x_subset[subset_index] /= 255

                y_subset[subset_index] = y_set[random_index]

                subset_index += 1

            except:
                print("Ignore image {}".format(x_set[random_index]))

        return x_subset, keras.utils.to_categorical(y_subset, LeapGestRecog.num_classes)


# if __name__== "__main__":
#     training_folders = ['00', '01', '02', '03', '04', '05', '06', '07', '08']
#     testing_folders = ['09']
#
#     training_images = []
#     training_labels = []
#
#     testing_images = []
#     testing_labels = []
#
#     for folder in training_folders:
#         x_set, y_set = LeapGestRecog.read_data_set_image_paths(os.path.join(LeapGestRecog.data_set_path, folder))
#         training_images.extend(x_set)
#         training_labels.extend(y_set)
#
#     for folder in testing_folders:
#         x_set, y_set = LeapGestRecog.read_data_set_image_paths(os.path.join(LeapGestRecog.data_set_path, folder))
#         testing_images.extend(x_set)
#         testing_labels.extend(y_set)
#
#     print(testing_images)
#     print(testing_labels)
#