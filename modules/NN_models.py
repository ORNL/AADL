from abc import ABCMeta, abstractmethod, ABC
import torch
import numpy


def activation_function(name_activation):
    if name_activation == 'relu':
        return torch.nn.ReLU()
    elif name_activation == 'sigmoid':
        return torch.nn.Sigmoid()
    elif name_activation == 'softmax':
        return torch.nn.Softmax()
    elif name_activation == 'tanh':
        return torch.nn.Tanh()
    else:
        raise ValueError("Activation function not recognized")


# This is an ABSTRACT class
class NeuralNetwork(torch.nn.Module, metaclass=ABCMeta):

    def __init__(self, input_dim: int, output_dim: int, num_neurons_list: list, use_bias: bool, activation: str,
                 classification: bool, device='cpu'):
        """

        :param input_dim: :type int
        :param output_dim: :type int
        :param num_neurons_list: :type list
        :param use_bias: :type bool
        :param activation: :type str
        :param classification: :type bool
        """
        super().__init__()

        self.device = torch.device(device)

        assert isinstance(input_dim, int) or isinstance(input_dim, tuple)
        self.input_dim = input_dim

        assert isinstance(output_dim, int) or isinstance(output_dim, tuple)
        self.output_dim = output_dim

        assert isinstance(num_neurons_list, list)
        assert len(num_neurons_list) > 0
        self.num_hidden_layers = len(num_neurons_list)
        self.num_neurons_list = num_neurons_list

        assert isinstance(use_bias, bool)
        self.use_bias = use_bias

        self.model = None
        self.layers = []

        assert isinstance(activation, str)
        self.activation = activation

        assert isinstance(classification, bool)
        self.classification = classification

    def forward(self, input_data):
        """

        :param input_data: :type: float
        """
        x = input_data
        y = self.model(x.float())
        y = torch.squeeze(y, 1)

        return y

    # getter method for model
    def get_model(self):
        return self.model

    # getter method for device
    def get_device(self):
        return self.device

    # extract list of layers
    def extract_layers_in_list(self):
        layers = [module for module in self.model.modules() if type(module) != torch.nn.Sequential]
        return layers

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


class MLP(NeuralNetwork, ABC):

    def __init__(self, input_dim: int, output_dim: int, num_neurons_list: list, use_bias: bool, activation: str,
                 classification=False, device='cpu'):
        """

        :param input_dim: :type int
        :param output_dim: :type int
        :param num_neurons_list: :type list
        :param use_bias: :type bool
        """
        super(MLP, self).__init__(input_dim, output_dim, num_neurons_list, use_bias, activation, classification, device)

        # Input layer
        self.layers += [torch.nn.Linear(self.input_dim, self.num_neurons_list[0], bias=self.use_bias)]
        self.layers += [activation_function(self.activation)]

        # Hidden layers
        for layer_index in range(0, len(self.num_neurons_list) - 1):
            self.layers += [torch.nn.Linear(self.num_neurons_list[layer_index], self.num_neurons_list[layer_index + 1],
                                            bias=self.use_bias)]
            self.layers += [activation_function(self.activation)]

        # Output layer
        self.layers += [torch.nn.Linear(self.num_neurons_list[-1], self.output_dim, bias=self.use_bias)]

        # Activation function for classification problem
        if classification:
            self.layers += [torch.nn.Softmax()]

        # Multilayer perceptron
        self.model = torch.nn.Sequential(*self.layers)


class CNN2D(NeuralNetwork, ABC):

    def __init__(self, input_dim: int, output_dim: int, num_neurons_list: list, use_bias: bool, activation: str,
                 classification=False, device='cpu', **kwargs):
        """

        :param input_dim: :type int
        :param output_dim: :type int
        :param num_neurons_list: :type list
        :param use_bias: :type bool
        """
        super(CNN2D, self).__init__(input_dim, output_dim, num_neurons_list, use_bias, activation, classification, device,
                                    **kwargs)

        self.kernel_size_list = kwargs.get("kernel_size_list", None)
        self.max_pooling_list = kwargs.get("max_pooling_list", None)
        self.batch_norm_list = kwargs.get("batch_norm_list", None)
        self.padding_list = kwargs.get("padding_list", None)
        self.stride_list = kwargs.get("stride_list", None)
        self.input_channels = input_dim[0]
        self.image_size = numpy.prod(input_dim[1:])

        if self.kernel_size_list is not None:
            assert isinstance(self.kernel_size_list, list)
            assert len(self.num_neurons_list) == len(self.kernel_size_list)

        if self.max_pooling_list is not None:
            assert isinstance(self.max_pooling_list, list)
            assert len(self.num_neurons_list) == len(self.max_pooling_list)

        if self.batch_norm_list is not None:
            assert isinstance(self.batch_norm_list, list)
            assert len(self.batch_norm_list) == len(self.batch_norm_list)

        if self.padding_list is not None:
            assert isinstance(self.padding_list, list)
            assert len(self.padding_list) == len(self.padding_list)

        if self.stride_list is not None:
            assert isinstance(self.stride_list, list)
            assert len(self.stride_list) == len(self.stride_list)

        if self.kernel_size_list is not None:
            ker_size = self.kernel_size_list[0]
        else:
            ker_size = 3
        self.layers += [torch.nn.Conv2d(in_channels=self.input_channels, out_channels=self.num_neurons_list[0],
                                        kernel_size=ker_size, stride=1, padding=1,
                                        bias=self.use_bias)]

        # Hidden layers
        for layer_index in range(0, len(self.num_neurons_list) - 1):
            if self.kernel_size_list is not None:
                ker_size = self.kernel_size_list[layer_index]
            else:
                ker_size = 3

            self.layers += [torch.nn.Conv2d(in_channels=self.num_neurons_list[layer_index],
                                            out_channels=self.num_neurons_list[layer_index + 1],
                                            kernel_size=ker_size, stride=1, padding=1,
                                            bias=self.use_bias)]
            self.layers += [activation_function(self.activation)]

        # Output layer
        self.layers += [torch.nn.Flatten()]
        self.layers += [
            torch.nn.Linear(in_features=self.image_size * self.num_neurons_list[-1], out_features=self.output_dim,
                            bias=self.use_bias)]

        # Activation function for classification problem
        self.layers += [torch.nn.Softmax()]

        # Convolutional neural network
        self.model = torch.nn.Sequential(*self.layers)
