from abc import ABC

import torch
import numpy


class MultiLayerPerceptionRegression(object):

    def __init__(self, input_dim: int, output_dim: int, num_neurons: int, num_hidden_layers: int, use_bias: bool):
        """

        :type input_dim: int
        :type output_dim: int
        :type num_neurons: int
        :type num_hidden_layers: int
        """
        super(MultiLayerPerceptionRegression, self).__init__()

        self.device = torch.device('cpu')

        assert isinstance(input_dim, int)
        self.input_dim = input_dim

        assert isinstance(output_dim, int)
        self.output_dim = output_dim

        assert isinstance(num_hidden_layers, int)
        self.num_hidden_layers = num_hidden_layers

        assert isinstance(num_neurons, int)
        self.num_neurons_per_layer = num_neurons

        assert isinstance(use_bias, bool)
        self.use_bias = use_bias

        self.model = None
        self.layers = []
        self.training_loss_history = []
        self.loss_function_specified = False
        self.criterion = None

        # Input layer
        self.layers += [torch.nn.Linear(self.input_dim, self.num_neurons_per_layer, bias=self.use_bias)]
        self.layers += [torch.nn.ReLU()]

        # Hidden layers
        for hidden_layer_index in range(0, self.num_hidden_layers):
            self.layers += [torch.nn.Linear(self.num_neurons_per_layer, self.num_neurons_per_layer, bias=self.use_bias)]
            self.layers += [torch.nn.ReLU()]

        # Output layer
        self.layers += [torch.nn.Linear(self.num_neurons_per_layer, self.output_dim, bias=self.use_bias)]

        self.model = torch.nn.Sequential(*self.layers)

    # Alternative definition of the class where a specific number of neuron per layer is passed
    """
        def __init__(self, input_dim: int, output_dim: int, num_neurons_list: list):

            super(MultiLayerPerceptionRegression, self).__init__()

            self.device = torch.device('cpu')

            assert isinstance(input_dim, int)
            self.input_dim = input_dim

            assert isinstance(output_dim, int)
            self.output_dim = output_dim

            assert isinstance(num_neurons_list, int)
            assert len(num_neurons_list) > 0
            self.num_hidden_layers = len(num_neurons_list)
            self.num_neurons_list = num_neurons_list

            self.model = None
            self.layers = []

            # Input layer
            self.layers += [torch.nn.Linear(self.input_dim, self.num_neurons_list[0])]
            self.layers += [torch.nn.ReLU()]

            # Hidden layers
            for layer_index in range(0,len(self.num_neurons_list)-1):
                self.layers += [torch.nn.Linear(self.num_neurons_list[layer_index], self.num_neurons_list[layer_index+1])]
                self.layers += [torch.nn.ReLU()]

            # Output layer
            self.layers += [torch.nn.Linear(self.num_neurons_list[-1], self.output_dim)]

            self.model = torch.nn.Sequential(*self.layers)
    """

    def evaluate(self, input_data):

        assert (input_data.shape[1] == self.input_dim)

        # Before using the neural network for predictions, we need to specify that the learning has finished
        self.model.train(mode=False)

        x = input_data
        y = self.model(x.float())

        return y

    def to(self, device):
        """

        :type device: torch.device
        """
        assert isinstance(device, torch.device)
        self.model.to(device)
        self.device = device

    # getter method for model
    def get_model(self):
        return self.model


    def set_coefficients_to_zero(self):

        layers = [module for module in self.model.modules() if type(module) != torch.nn.Sequential]

        for layer in layers:
            if isinstance(layer, torch.nn.modules.linear.Linear):
                torch.nn.init.zeros_(layer.weight)

                if self.use_bias:
                    torch.nn.init.zeros_(layer.bias)

    def set_coefficients_to_one(self):

        layers = [module for module in self.model.modules() if type(module) != torch.nn.Sequential]

        for layer in layers:
            if isinstance(layer, torch.nn.modules.linear.Linear):
                torch.nn.init.ones_(layer.weight)

                if self.use_bias:
                    torch.nn.init.ones_(layer.bias)

    def set_coefficients_to_random(self):

        layers = [module for module in self.model.modules() if type(module) != torch.nn.Sequential]

        for layer in layers:
            if isinstance(layer, torch.nn.modules.linear.Linear):
                torch.nn.init.normal_(layer.weight)

                if self.use_bias:
                    torch.nn.init.normal_(layer.bias)


    def print_architecture(self):  # static method
        print(self.model)

    def extract_layers_in_list(self):
        layers = [module for module in self.model.modules() if type(module) != torch.nn.Sequential]
        return layers
