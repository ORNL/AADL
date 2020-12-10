import sys
import numpy
import torch
from torch.utils.data import Dataset

import unittest

sys.path.append('../utils')
sys.path.append('../modules')
import dataloaders
from NN_models import MLP
from optimizers import FixedPointIteration, RNA_Acceleration


###############################################################################


def test_neural_network_linear_regression(slope, intercept, num_points, optimizer_str):
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

    dataloader = torch.utils.data.DataLoader(dataset, batch_size)

    model = MLP(inputDim, outputDim, num_neurons_list, use_bias, activation, classification_problem)

    optimizer_classic = FixedPointIteration(dataloader, learning_rate, weight_decay, )
    optimizer_classic.import_model(model)
    optimizer_classic.set_loss_function('mse')
    optimizer_classic.set_optimizer(optimizer_str)
    training_classic_loss_history = optimizer_classic.train(epochs, threshold, batch_size)

    weights = list(model.get_model().parameters())

    return weights, training_classic_loss_history


###############################################################################


class TestLinearRegression(unittest.TestCase):

    def test_neural_network_linear_regression_sgd(self):
        num_points = 2
        straight_line_parameters = torch.rand(2, 1)
        slope = straight_line_parameters[0].item()
        intercept = straight_line_parameters[1].item()
        numeric_weights, history = test_neural_network_linear_regression(slope, intercept, num_points, 'sgd')
        self.assertTrue( history[-1].item()<1e-8 )
    
    def test_neural_network_linear_regression_rmsprop(self):
        num_points = 2
        straight_line_parameters = torch.rand(2, 1)
        slope = straight_line_parameters[0].item()
        intercept = straight_line_parameters[1].item()
        numeric_weights, history = test_neural_network_linear_regression(slope, intercept, num_points, 'rmsprop')
        self.assertTrue( history[-1].item()<1e-8 )
    
    def test_neural_network_linear_regression_adam(self):
        num_points = 2
        straight_line_parameters = torch.rand(2, 1)
        slope = straight_line_parameters[0].item()
        intercept = straight_line_parameters[1].item()
        numeric_weights, history = test_neural_network_linear_regression(slope, intercept, num_points, 'adam')
        self.assertTrue( history[-1].item()<1e-8 )      


###############################################################################


if __name__ == "__main__":
    unittest.main()

