import time
import torch
import numpy
from torch import Tensor
from torch import autograd
from abc import ABCMeta, abstractmethod, ABC
import math


# Abstract class that provides basic guidelines to implement an acceleration
class Acceleration(object, metaclass=ABCMeta):
    def __init__(self, window_depth: int, frequency: int, learning_rate: float, weight_decay: float):
        """

        :type window_depth: int
        :type frequency: int
        :type learning_rate: float
        """
        self.iteration_counter = 0

        assert isinstance(learning_rate, float)
        self.lr = learning_rate

        assert isinstance(weight_decay, float)
        self.weight_decay = weight_decay

        assert isinstance(window_depth, int)
        self.window_depth = window_depth

        assert isinstance(frequency, int)
        self.freq = frequency

        self.model_imported = False
        self.model = None

        self.training_loss_history = []
        self.criterion_specified = False
        self.criterion = None
        self.optimizer_specified = False
        self.optimizer = None

    def set_zero_grad(self):
        assert self.model_imported
        torch.autograd.zero_grad(self.model.parameters())

    def import_model(self, model):
        assert not self.model_imported
        assert isinstance(model, object)
        self.model = model
        self.model_imported = True

    def get_model(self):
        assert self.model_imported
        return self.model

    @abstractmethod
    def accelerated_train(self, input_data: torch.Tensor, target: torch.Tensor, num_iterations: int, threshold: float,
                          batch_size: int):
        pass

    def set_loss_function(self, criterion_string):

        if criterion_string.lower() == 'mse':
            self.criterion = torch.nn.MSELoss()
            self.criterion_specified = True
        elif criterion_string.lower() == 'bce':
            self.criterion = torch.nn.BCELoss()
            self.criterion_specified = True
        else:
            raise ValueError("Loss function is not recognized: currently only MSE and BCE are allowed")

    @property
    def is_loss_function_set(self):
        return self.criterion_specified

    def set_optimizer(self, optimizer_string):

        # we will need the parameters of the deep learning model as inout for the torch.optim object
        # so first we need to make sure that we have already imported the neural network
        assert self.model_imported

        if optimizer_string.lower() == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.get_model().parameters(), lr=self.lr, weight_decay=self.weight_decay)
            self.optimizer_specified = True
        elif optimizer_string.lower() == 'adam':
            self.optimizer = torch.optim.Adam(self.model.get_model().parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=self.weight_decay)
            self.optimizer_specified = True
        else:
            raise ValueError("Optimizer is not recognized: currently only SGD and Adam are allowed")

    @property
    def is_optimizer_set(self):
        return self.optimizer_specified


class Anderson(Acceleration, ABC):
    def __init__(self, window_depth: int, frequency: int, learning_rate: float, weight_decay: float):
        """

        :type window_depth: int
        :type frequency: int
        :type learning_rate: float
        """
        super(Anderson, self).__init__(window_depth, frequency, learning_rate, weight_decay)

    def accelerated_train(self, input_data, target, num_epochs, threshold, batch_size):
        assert self.optimizer_specified
        assert batch_size < input_data.shape[0]

        # Define the objective function to optimize during the training of the neural network
        loss = self.criterion(input_data, target)

        epoch_counter = 0

        permutation = torch.randperm(input_data.size()[0])

        while (loss.item() > threshold) & (epoch_counter < num_epochs):

            print("###############################")
            print('Epoch: ' + str(epoch_counter) + ' - Loss function: ' + str(loss.item()))

            for i in range(0, input_data.shape[0], batch_size):
                indices = permutation[i:i + batch_size]
                batch_x, batch_y = input_data[indices], target[indices]
                self.optimizer.zero_grad()  # zero the gradient buffers
                output = self.model.get_model()(batch_x)
                loss = self.criterion(output, batch_y)
                loss.backward()
                self.optimizer.step()  # Does the update

            epoch_counter = epoch_counter + 1
            self.training_loss_history.append(loss)

        return self.training_loss_history
