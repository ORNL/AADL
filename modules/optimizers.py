import time
import torch
import numpy
from torch import Tensor
from torch import autograd
from abc import ABCMeta, abstractmethod, ABC
import math
from AccelerationModule import AccelerationModule


# Abstract class that provides basic guidelines to implement an acceleration
class Optimizer(object, metaclass=ABCMeta):
    def __init__(self, data_loader: torch.utils.data.dataloader.DataLoader, learning_rate: float,
                 weight_decay: float = 0.0, verbose: bool = False):
        """

        :type data_loader: torch.utils.data.dataloader.DataLoader
        :type learning_rate: float
        :type weight_decay: float
        """
        self.iteration_counter = 0

        assert isinstance(data_loader, torch.utils.data.dataloader.DataLoader)
        self.data_loader = data_loader

        assert isinstance(learning_rate, float)
        self.lr = learning_rate

        assert isinstance(weight_decay, float)
        self.weight_decay = weight_decay

        self.model_imported = False
        self.model = None

        self.training_loss_history = []
        self.criterion_specified = False
        self.criterion = None
        self.optimizer_specified = False
        self.optimizer = None
        self.loss_name = None

        self.verbose = verbose

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
    def train(self, input_data: torch.Tensor, target: torch.Tensor, num_iterations: int, threshold: float,
              batch_size: int):
        pass

    def set_loss_function(self, criterion_string):

        if criterion_string.lower() == 'mse':
            self.criterion = torch.nn.MSELoss()
            self.criterion_specified = True
        elif criterion_string.lower() == 'ce':
            self.criterion = torch.nn.CrossEntropyLoss()
            self.criterion_specified = True
        else:
            raise ValueError("Loss function is not recognized: currently only MSE and CE are allowed")
        self.loss_name = criterion_string

    @property
    def is_loss_function_set(self):
        return self.criterion_specified

    def set_optimizer(self, optimizer_string):

        # we will need the parameters of the deep learning model as inout for the torch.optim object
        # so first we need to make sure that we have already imported the neural network
        assert self.model_imported

        if optimizer_string.lower() == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.get_model().parameters(), lr=self.lr,
                                             weight_decay=self.weight_decay)
            self.optimizer_specified = True
        elif optimizer_string.lower() == 'rmsprop':
            self.optimizer = torch.optim.RMSprop(self.model.get_model().parameters(), lr=self.lr, alpha=0.99,
                                                 weight_decay=self.weight_decay)
            self.optimizer_specified = True
        elif optimizer_string.lower() == 'adam':
            self.optimizer = torch.optim.Adam(self.model.get_model().parameters(), lr=self.lr, betas=(0.9, 0.999),
                                              weight_decay=self.weight_decay)
            self.optimizer_specified = True
        else:
            raise ValueError("Optimizer is not recognized: currently only SGD, RMSProp and Adam are allowed")

    @property
    def is_optimizer_set(self):
        return self.optimizer_specified

    def print_verbose(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)


class FixedPointIteration(Optimizer, ABC):
    def __init__(self, data_loader: torch.utils.data.dataloader.DataLoader, learning_rate: float,
                 weight_decay: float = 0.0, verbose: bool = False):
        """

        :param learning_rate: :type: float
        :param weight_decay: :type: float
        """
        super(FixedPointIteration, self).__init__(data_loader, learning_rate, weight_decay, verbose)

    def train(self, num_epochs, threshold, batch_size):

        self.model.get_model().train(True)  # True indicates actual training

        assert self.optimizer_specified
        epoch_counter = 0
        value_loss = float('Inf')

        while epoch_counter < num_epochs and value_loss > threshold:

            for batch_idx, (data, target) in enumerate(self.data_loader):
                data, target = data.to(self.model.get_device()), target.to(self.model.get_device())
                self.optimizer.zero_grad()
                output = self.model.forward(data)
                # print("Input_data: "+str(data.shape)+' - Output: '+str(output.shape)+' - Target: '+str(target.shape)+' - '+' - Value:'+str(output))
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()

            self.print_verbose("###############################")
            self.print_verbose('Epoch: ' + str(epoch_counter) + ' - Loss function: ' + str(loss.item()))

            value_loss = loss.item()
            epoch_counter = epoch_counter + 1
            self.training_loss_history.append(loss)

        return self.training_loss_history


class DeterministicAcceleration(Optimizer, ABC):
    def __init__(self, data_loader: torch.utils.data.dataloader.DataLoader, acceleration_type: str, learning_rate: float, relaxation:float, 
                 weight_decay: float = 0.0, wait_iterations: int = 1, history_depth: int = 15, frequency: int = 1,
                 reg_acc: float = 0.0, store_each: int = 1, verbose: bool = False):
        """

        :param learning_rate: :type: float
        :param weight_decay: :type: float
        """
        super(DeterministicAcceleration, self).__init__(data_loader, learning_rate, weight_decay, verbose)
        self.acceleration_type = acceleration_type.lower()
        self.wait_iterations = wait_iterations
        self.relaxation = relaxation
        self.store_each = store_each
        self.history_depth = history_depth
        self.frequency = frequency
        self.reg_acc = reg_acc

    def train(self, num_epochs, threshold, batch_size):

        assert self.model_imported

        # Initialization of acceleration module
        self.acc_mod = AccelerationModule(self.acceleration_type, self.model.get_model(), self.history_depth, self.reg_acc, self.store_each)
        self.acc_mod.store(self.model.get_model())

        self.model.get_model().train(True)
        assert self.optimizer_specified
        epoch_counter = 0
        value_loss = float('Inf')

        while epoch_counter < num_epochs and value_loss > threshold:

            for batch_idx, (data, target) in enumerate(self.data_loader):
                data, target = data.to(self.model.get_device()), target.to(self.model.get_device())
                self.optimizer.zero_grad()
                output = self.model.forward(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()

            self.print_verbose("###############################")
            self.print_verbose('Epoch: ' + str(epoch_counter) + ' - Loss function: ' + str(loss.item()))

            # Acceleration
            self.acc_mod.store(self.model.get_model())
            if (epoch_counter > self.wait_iterations) and (epoch_counter % self.frequency == 0):
                self.acc_mod.accelerate(self.model.get_model(), self.relaxation)

            value_loss = loss.item()
            epoch_counter = epoch_counter + 1
            self.training_loss_history.append(loss)

        return self.training_loss_history
