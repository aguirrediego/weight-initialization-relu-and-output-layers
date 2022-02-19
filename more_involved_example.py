from experiment_definition.experiment_definition import *
from utils.utils import Utils
from initialization_techniques.initialization_techniques_manager import InitializationTechniqueOptions, \
                                                                        InitializationTechniqueManager
from initialization_techniques.scheme1_init import WeightInitScheme1Params
from initialization_techniques.scheme4_init import WeightInitScheme4Params
from initialization_techniques.he_init import WeightInitHeParams
from initialization_techniques.glorot_init import WeightInitGlorotParams
from initialization_techniques.lsuv_init import WeightInitLSUVParams

from models.model_manager import ModelOptions, ModelManager
from data_sets.data_set_manager import DataSetOptions, DataSetManager
import time
import keras
import os
import pickle
import hashlib 
import math
import tensorflow as tf


class ExperimentOnBatchesRunner:

    @staticmethod
    def run_experiment(experiment_obj=None, experiment_save_file_path=None):

        if experiment_obj is None:
            experiment_obj = ExperimentOnBatchesRunner.create_default_experiment_obj()

        if experiment_save_file_path is None:
            experiment_save_file_path = 'main/'

        experiment_str = str(experiment_obj)

        print(experiment_str)

        hash_object = hashlib.sha256(experiment_str.encode('utf-8'))
        hashed_experiment_str = hash_object.hexdigest()

        folder_path = experiment_save_file_path + '/' + hashed_experiment_str

        if os.path.isdir(folder_path):
            return

        input_shape = DataSetManager.get_input_shape(experiment_obj.data_set)
        num_classes = DataSetManager.get_num_classes(experiment_obj.data_set)

        model = ModelManager.get_model(experiment_obj.model, input_shape, num_classes)

        init_technique_hidden_layers = \
            InitializationTechniqueManager.get_initialization_technique(experiment_obj.init_method_hidden_layers)

        init_technique_output_layers = \
            InitializationTechniqueManager.get_initialization_technique(experiment_obj.init_method_output_layers)

        init_technique_hidden_layers_params = \
            ExperimentOnBatchesRunner.resolve_init_params(experiment_obj.init_method_hidden_layers,
                                                 experiment_obj.init_method_hidden_layers_params,
                                                 experiment_obj.data_set, experiment_obj.verbose,
                                                 experiment_obj.random_seed, True)

        init_technique_output_layers_params = \
            ExperimentOnBatchesRunner.resolve_init_params(experiment_obj.init_method_output_layers,
                                                 experiment_obj.init_method_output_layers_params,
                                                 experiment_obj.data_set, experiment_obj.verbose,
                                                 experiment_obj.random_seed, False)

        model = init_technique_hidden_layers.initialize(model, init_technique_hidden_layers_params)
        model = init_technique_output_layers.initialize(model, init_technique_output_layers_params)

        # Initiate Stochastic Gradient Descent optimizer
        opt = tf.keras.optimizers.SGD(learning_rate=experiment_obj.learning_rate)
        # opt = keras.optimizers.SGD(learning_rate=experiment_obj.learning_rate)

        model.compile(loss='categorical_crossentropy',
                      optimizer=opt,
                      metrics=['accuracy'])

        num_test_examples = DataSetManager.get_num_testing_examples(experiment_obj.data_set)

        x_test, y_test = \
            DataSetManager.get_random_testing_subset(
                num_test_examples, experiment_obj.random_seed, experiment_obj.data_set)

        num_test_examples_per_batch = 100
        num_test_iterations = num_test_examples // num_test_examples_per_batch

        loss_sum = 0
        acc_sum = 0
        for i in range(num_test_iterations):

            x_test_mini_batch = x_test[i * num_test_examples_per_batch: (i + 1) * num_test_examples_per_batch]
            y_test_mini_batch = y_test[i * num_test_examples_per_batch: (i + 1) * num_test_examples_per_batch]

            loss_and_acc = model.test_on_batch(x_test_mini_batch, y_test_mini_batch)
            loss_sum += loss_and_acc[0]
            acc_sum += loss_and_acc[1]

        val_loss = [loss_sum / num_test_iterations]
        val_acc = [acc_sum / num_test_iterations]

        print("Initial Loss: ", val_loss)
        print("Initial Acc: ", val_acc)

        num_train_examples = DataSetManager.get_num_training_examples(experiment_obj.data_set)
        num_mini_batches = math.ceil(num_train_examples / experiment_obj.mini_batch_size)

        for k in range(experiment_obj.num_epochs):
            for j in range(num_mini_batches):
                start_time = time.time()

                x_mini_batch, y_mini_batch = \
                    DataSetManager.get_random_training_subset(experiment_obj.mini_batch_size,
                                                              None, experiment_obj.data_set)

                loss, accuracy = model.train_on_batch(x_mini_batch, y_mini_batch)
                end_time = time.time()

                print('epoch {}/{} batch {}/{} loss: {} accuracy: {} time: {}ms'.format(k + 1,
                            experiment_obj.num_epochs, j + 1, num_mini_batches, loss, accuracy,
                            1000 * (end_time - start_time)), flush=True)

            loss_sum = 0
            acc_sum = 0
            for i in range(num_test_iterations):
                x_test_mini_batch = x_test[i * num_test_examples_per_batch: (i + 1) * num_test_examples_per_batch]
                y_test_mini_batch = y_test[i * num_test_examples_per_batch: (i + 1) * num_test_examples_per_batch]

                loss_and_acc = model.test_on_batch(x_test_mini_batch, y_test_mini_batch)
                loss_sum += loss_and_acc[0]
                acc_sum += loss_and_acc[1]

            val_loss.append(loss_sum / num_test_iterations)
            val_acc.append(acc_sum / num_test_iterations)

            print("val_loss:", val_loss)
            print("val_acc:", val_acc)

        print("Done training")
        print("Saving results in:", folder_path)
        ExperimentOnBatchesRunner.save_experiment_results(experiment_obj, val_loss, val_acc, folder_path)
        print("Done")

    @staticmethod
    def save_experiment_results(experiment_obj, val_loss, val_acc, folder_path):

        if not os.path.isdir(folder_path):
            os.makedirs(folder_path)

        with open(folder_path + '/val_loss.pickle', 'wb') as val_loss_output:
            pickle.dump(val_loss, val_loss_output, protocol=pickle.HIGHEST_PROTOCOL)

        with open(folder_path + '/val_acc.pickle', 'wb') as val_acc_output:
            pickle.dump(val_acc, val_acc_output, protocol=pickle.HIGHEST_PROTOCOL)

        with open(folder_path + '/experiment_obj.pickle', 'wb') as experiment_obj_output:
            pickle.dump(experiment_obj, experiment_obj_output, protocol=pickle.HIGHEST_PROTOCOL)

        with open(folder_path + '/summary.txt', 'w+') as summary_output:
            summary_output.write(str(experiment_obj))
            summary_output.write("\nval_loss: " + str(val_loss))
            summary_output.write("\nval_acc: " + str(val_acc))

    @staticmethod
    def resolve_init_params(init_method, init_method_params, data_set, verbose, random_seed, init_hidden):
        if init_method == InitializationTechniqueOptions.HE:
            return WeightInitHeParams(verbose, random_seed, init_hidden)

        elif init_method == InitializationTechniqueOptions.GLOROT:
            return WeightInitGlorotParams(verbose, random_seed, init_hidden)

        init_set_size = init_method_params.init_set_size
        use_kmeans = False  # init_method_params.use_kmeans

        x_init_set, y_init_set = \
            ExperimentOnBatchesRunner.get_initialization_data_set(data_set, init_set_size, use_kmeans, random_seed)

        params = None

        if init_method == InitializationTechniqueOptions.SCHEME1:
            params = WeightInitScheme1Params(x_init_set, init_method_params.use_gram_schmidt, verbose,
                                             init_method_params.active_frac, init_method_params.goal_std)

        elif init_method == InitializationTechniqueOptions.SCHEME4:
            params = WeightInitScheme4Params(x_init_set, y_init_set, verbose)

        elif init_method == InitializationTechniqueOptions.LSUV:
            params = WeightInitLSUVParams(x_init_set, verbose,
                                          init_method_params.margin, init_method_params.max_iter, init_hidden)

        return params

    @staticmethod
    def create_default_experiment_obj():
        verbose = True
        model = ModelOptions.SIMPLE_CNN
        data_set = DataSetOptions.CIFAR10
        random_seed = 4
        learning_rate = 0.01
        mini_batch_size = 128
        num_epochs = 3
        init_method_hidden_layers = InitializationTechniqueOptions.SCHEME1

        init_method_hidden_layers_params = WeightInitScheme1Experiment(
            init_set_size=2048,
            use_kmeans=False,
            use_gram_schmidt=False,
            active_frac=0.8,
            goal_std=1.0
        )

        init_method_output_layers = InitializationTechniqueOptions.SCHEME4
        init_method_output_layers_params = WeightInitScheme4Experiment(
            init_set_size=2048,
            use_kmeans=False,
            pseudo_inverse_algo=None
        )

        experiment_obj = ExperimentParameters(verbose, model, data_set, random_seed,
                                              learning_rate, mini_batch_size, num_epochs,
                                              init_method_hidden_layers, init_method_hidden_layers_params,
                                              init_method_output_layers, init_method_output_layers_params)

        return experiment_obj

    @staticmethod
    def get_initialization_data_set(data_set, init_set_size, use_kmeans, random_seed):

        if use_kmeans:
            x_train, y_train, x_test, y_test = DataSetManager.get_data_set(data_set)
            x_init_set, y_init_set = Utils.create_init_set_kmeans(x_train, y_train, init_set_size, random_seed)
        else:
            x_init_set, y_init_set = DataSetManager.get_random_training_subset(init_set_size, random_seed, data_set)

        return x_init_set, y_init_set


def main():

    experiment = ExperimentOnBatchesRunner.create_default_experiment_obj()

    ExperimentOnBatchesRunner.run_experiment(experiment_obj=experiment, experiment_save_file_path='main/')


if __name__ == '__main__':
    main()
