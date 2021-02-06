import sys
import numpy
import torch
from torch.utils.data import Dataset

import unittest

sys.path.append('../utils')
sys.path.append('../modules')
import dataloaders
from NN_models import MLP
from optimizers import FixedPointIteration, DeterministicAcceleration


###############################################################################


def neural_network_linear_regression(slope, intercept, num_points, optimizer_str):
    inputDim, outputDim, dataset = dataloaders.linear_data(slope, intercept, num_points)
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
    inputDim, outputDim, dataset = dataloaders.linear_data(slope, intercept, num_points)
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
