import sys
import numpy
import torch
from torch.utils.data import Dataset

import unittest

sys.path.append('../utils')
from optimizers import FixedPointIteration, DeterministicAcceleration
sys.path.append('../model_zoo')
from NN_models import MLP


###############################################################################


def linear_regression_points(slope, intercept, n: int = 10):
    # create dummy data for training
    x_values = numpy.linspace(-10.0, 10.0, num=n)
    x_train = numpy.array(x_values, dtype=numpy.float32)
    x_train = x_train.reshape(-1, 1)

    y_values = [slope * i + intercept for i in x_values]
    y_train = numpy.array(y_values, dtype=numpy.float32)
    y_train = y_train.reshape(-1, 1)

    return x_train, y_train


class LinearData(Dataset):
    def __init__(self, slope, intercept, num_points: int = 10):
        super(LinearData, self).__init__()

        self.slope = slope
        self.intercept = intercept
        self.num_points = num_points

        x_sample, y_sample = linear_regression_points(self.slope, self.intercept, self.num_points)

        self.x_sample = x_sample
        self.y_values = y_sample
        self.y_values = numpy.reshape(self.y_values, (len(self.y_values), 1))

    def __len__(self):
        return self.y_values.shape[0]

    def __getitem__(self, index):
        x_sample = self.x_sample[index, :]

        y_sample = self.y_values[index]

        # Doubles must be converted to Floats before passing them to a neural network model
        x_sample = torch.from_numpy(x_sample).float()
        y_sample = torch.from_numpy(y_sample).float()

        return x_sample, y_sample


def linear_data(slope, intercept, num_points: int = 10):
    input_dim = 1
    output_dim = 1
    return input_dim, output_dim, LinearData(slope, intercept, num_points=num_points)


###############################################################################


def neural_network_linear_regression(slope, intercept, num_points, optimizer_str):
    inputDim, outputDim, dataset = linear_data(slope, intercept, num_points)
    num_neurons_list = [1]
    use_bias = True
    classification_problem = False
    activation = None
    weight_decay = 0.0
    learning_rate = 1e-3
    batch_size = 1
    epochs = 10000
    threshold = 1e-8

    training_dataloader = torch.utils.data.DataLoader(dataset, batch_size)
    validation_dataloader = torch.utils.data.DataLoader(dataset, batch_size)

    model = MLP(inputDim,outputDim,num_neurons_list,use_bias,activation,classification_problem)

    optimizer_classic = FixedPointIteration(training_dataloader, validation_dataloader,learning_rate, weight_decay)
    optimizer_classic.import_model(model)
    optimizer_classic.set_loss_function('mse')
    optimizer_classic.set_optimizer(optimizer_str)
    training_classic_loss_history, validation_classic_loss_history, _ = optimizer_classic.train(epochs, threshold, batch_size)

    weights = list(model.get_model().parameters())

    return weights, validation_classic_loss_history


def neural_network_linear_regression_anderson(slope, intercept, num_points, optimizer_str):
    inputDim, outputDim, dataset = linear_data(slope, intercept, num_points)
    num_neurons_list = [1]
    use_bias = True
    classification_problem = False
    activation = None
    weight_decay = 0.0
    learning_rate = 1e-3
    relaxation = 1e-1
    weight_decay = 0.0
    batch_size = 1
    epochs = 10000
    threshold = 1e-8
    wait_iterations = 1
    history_depth = 100
    frequency = 1
    reg_acc = 1e-9
    store_each_nth = 1

    training_dataloader = torch.utils.data.DataLoader(dataset, batch_size)
    validation_dataloader = torch.utils.data.DataLoader(dataset, batch_size)

    model = MLP(inputDim,outputDim,num_neurons_list,use_bias,activation,classification_problem)
    optimizer_anderson = DeterministicAcceleration(training_dataloader,validation_dataloader,'anderson',learning_rate,relaxation,weight_decay,wait_iterations,history_depth,frequency,reg_acc,store_each_nth)
    optimizer_anderson.import_model(model)
    optimizer_anderson.set_loss_function('mse')
    optimizer_anderson.set_optimizer(optimizer_str)
    training_anderson_loss_history, validation_anderson_loss_history, _ = optimizer_anderson.train(epochs, threshold, batch_size)

    weights = list(model.get_model().parameters())

    return weights, validation_anderson_loss_history


def test_neural_network_linear_regression(optimizer):
    num_points = 2
    straight_line_parameters = torch.rand(2, 1)
    slope = straight_line_parameters[0].item()
    intercept = straight_line_parameters[1].item()
    numeric_weights, history = neural_network_linear_regression(slope, intercept, num_points, optimizer)
    assert history[-1] < 1e-7


def test_neural_network_linear_regression_anderson(optimizer):
    num_points = 2
    straight_line_parameters = torch.rand(2, 1)
    slope = straight_line_parameters[0].item()
    intercept = straight_line_parameters[1].item()
    numeric_weights, history = neural_network_linear_regression_anderson(slope, intercept, num_points, optimizer)
    assert history[-1] < 1e-7


###############################################################################


class TestLinearRegression(unittest.TestCase):
    def test_neural_network_linear_regression_sgd(self):
        test_neural_network_linear_regression('sgd')

    def test_neural_network_linear_regression_rmsprop(self):
        test_neural_network_linear_regression('rmsprop')

    def test_neural_network_linear_regression_adam(self):
        test_neural_network_linear_regression('adam')

    def test_neural_network_linear_regression_sgd_anderson(self):
        test_neural_network_linear_regression_anderson('sgd')

    def test_neural_network_linear_regression_rmsprop_andserson(self):
        test_neural_network_linear_regression_anderson('rmsprop')

    def test_neural_network_linear_regression_adam_anderson(self):
        test_neural_network_linear_regression_anderson('adam')


###############################################################################


if __name__ == "__main__":
    unittest.main()
