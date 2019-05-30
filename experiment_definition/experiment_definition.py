class ExperimentParameters:
    def __init__(self, verbose, model, data_set, random_seed, learning_rate, mini_batch_size, num_epochs,
                 init_method_hidden_layers, init_method_hidden_layers_params,
                 init_method_output_layers, init_method_output_layers_params):
        self.verbose = verbose
        self.model = model
        self.data_set = data_set
        self.random_seed = random_seed
        self.learning_rate = learning_rate
        self.mini_batch_size = mini_batch_size
        self.num_epochs = num_epochs
        self.init_method_hidden_layers = init_method_hidden_layers
        self.init_method_hidden_layers_params = init_method_hidden_layers_params
        self.init_method_output_layers = init_method_output_layers
        self.init_method_output_layers_params = init_method_output_layers_params

    def __str__(self):

        s = "verbose: " + str(self.verbose)
        s += "\nmodel: " + str(self.model)
        s += "\ndata_set: " + str(self.data_set)
        s += "\nrandom_seed: " + str(self.random_seed)
        s += "\nlearning_rate: " + str(self.learning_rate)
        s += "\nmini_batch_size: " + str(self.mini_batch_size)
        s += "\nnum_epochs: " + str(self.num_epochs)
        s += "\ninit_method_hidden_layers: " + str(self.init_method_hidden_layers)
        s += "\ninit_method_hidden_layers_params: " + str(self.init_method_hidden_layers_params)
        s += "\ninit_method_output_layers: " + str(self.init_method_output_layers)
        s += "\ninit_method_output_layers_params: " + str(self.init_method_output_layers_params)

        return s


class WeightInitScheme1Experiment:
    def __init__(self, init_set_size, use_kmeans, use_gram_schmidt, active_frac, goal_std):
        self.init_set_size = init_set_size
        self.use_kmeans = use_kmeans
        self.use_gram_schmidt = use_gram_schmidt
        self.active_frac = active_frac
        self.goal_std = goal_std

    def __str__(self):

        s = "\n-init_set_size: " + str(self.init_set_size)
        s += "\n-use_kmeans: " + str(self.use_kmeans)
        s += "\n-use_gram_schmidt: " + str(self.use_gram_schmidt)
        s += "\n-active_frac: " + str(self.active_frac)
        s += "\n-goal_std: " + str(self.goal_std)
        return s


class WeightInitScheme4Experiment:
    def __init__(self, init_set_size, use_kmeans, pseudo_inverse_algo):
        self.init_set_size = init_set_size
        self.use_kmeans = use_kmeans
        self.pseudo_inverse_algo = pseudo_inverse_algo

    def __str__(self):

        s = "\n-init_set_size: " + str(self.init_set_size)
        s += "\n-use_kmeans: " + str(self.use_kmeans)
        s += "\n-pseudo_inverse_algo: " + str(self.pseudo_inverse_algo)

        return s


class WeightInitLSUVExperiment:
    def __init__(self, init_set_size, use_kmeans, margin, max_iter):
        self.init_set_size = init_set_size
        self.use_kmeans = use_kmeans
        self.margin = margin
        self.max_iter = max_iter

    def __str__(self):
        s = "\n-init_set_size: " + str(self.init_set_size)
        s += "\n-use_kmeans: " + str(self.use_kmeans)
        s += "\n-margin: " + str(self.margin)
        s += "\n-max_iter: " + str(self.max_iter)

        return s
