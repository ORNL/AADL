import sys
import torch

import unittest

sys.path.append('../utils')
sys.path.append('../modules')
import dataloaders
from LinearRegression_models import LinearRegression
from optimizers import FixedPointIteration, DeterministicAcceleration


###############################################################################


def linear_regression(slope, intercept, num_points, optimizer_str):
    input_dim, output_dim, dataset = dataloaders.linear_data(slope, intercept, num_points)
    use_bias = True
    learning_rate = 1e-3
    weight_decay = 0.0
    batch_size = 1
    epochs = 10000
    threshold = 1e-8

    training_dataloader = torch.utils.data.DataLoader(dataset, batch_size)
    validation_dataloader = torch.utils.data.DataLoader(dataset, batch_size)

    model = LinearRegression(input_dim, output_dim, use_bias)
    optimizer_classic = FixedPointIteration(training_dataloader, validation_dataloader, learning_rate, weight_decay)
    optimizer_classic.import_model(model)
    optimizer_classic.set_loss_function('mse')
    optimizer_classic.set_optimizer(optimizer_str)
    training_classic_loss_history, validation_classic_loss_history, _ = optimizer_classic.train(epochs, threshold, batch_size)

    weights = list(model.get_model().parameters())

    return weights[0].item(), weights[1].item(), validation_classic_loss_history


def linear_regression_anderson(slope, intercept, num_points, optimizer_str):
    input_dim, output_dim, dataset = dataloaders.linear_data(slope, intercept, num_points)
    use_bias = True
    learning_rate = 1e-3
    relaxation = 0.5
    weight_decay = 0.0
    batch_size = 1
    epochs = 10000
    threshold = 1e-8
    wait_iterations = 1
    history_depth = 3
    frequency = 1
    reg_acc = 1e-9
    store_each_nth = 1

    training_dataloader = torch.utils.data.DataLoader(dataset, batch_size)
    validation_dataloader = torch.utils.data.DataLoader(dataset, batch_size)

    model = LinearRegression(input_dim, output_dim, use_bias)
    optimizer_anderson = DeterministicAcceleration(training_dataloader,validation_dataloader,'anderson',learning_rate,relaxation,weight_decay,wait_iterations,history_depth,
        frequency,reg_acc,store_each_nth)
    optimizer_anderson.import_model(model)
    optimizer_anderson.set_loss_function('mse')
    optimizer_anderson.set_optimizer(optimizer_str)
    training_anderson_loss_history, validation_anderson_loss_history, _ = optimizer_anderson.train(epochs, threshold, batch_size)

    weights = list(model.get_model().parameters())

    return weights[0].item(), weights[1].item(), validation_anderson_loss_history


def test_linear_regression(optimiser):
    num_points = 2
    straight_line_parameters = torch.ones(2, 1)
    slope = straight_line_parameters[0].item()
    intercept = straight_line_parameters[1].item()
    numeric_slope, numeric_intercept, history = linear_regression(slope, intercept, num_points, optimiser)
    print(optimiser+" converged in "+ str(len(history))+" iterations "+"\n exact slope: "+ str(slope)+"  - "+ " numerical slope: "+str(numeric_slope)+"\n"+" exact intercept: "
        + str(intercept)+" - "+" numerical intercept: "+str(numeric_intercept))
    assert(abs((slope - numeric_slope)) < 1e-3 and abs((intercept - numeric_intercept)) < 1e-3)


def test_linear_regression_anderson(optimiser):
    num_points = 2
    straight_line_parameters = torch.ones(2, 1)
    slope = straight_line_parameters[0].item()
    intercept = straight_line_parameters[1].item()
    numeric_slope, numeric_intercept, history = linear_regression_anderson(
        slope, intercept, num_points, optimiser
    )
    print(optimiser+" + Anderson converged in "+ str(len(history))+" iterations "+"\n exact slope: "+ str(slope)+"  - "+ " numerical slope: "+str(numeric_slope)+"\n"+" exact intercept: "
        + str(intercept)+" - "+" numerical intercept: "+str(numeric_intercept))
    assert(abs((slope - numeric_slope)) < 1e-3 and abs((intercept - numeric_intercept)) < 1e-3)


###############################################################################


class TestLinearRegression(unittest.TestCase):
    
    def test_sgd(self):
        test_linear_regression('sgd')

    def test_rmsprop(self):
        test_linear_regression('rmsprop')

    def test_adam(self):
        test_linear_regression('adam')

    def test_sgd_anderson(self):
        test_linear_regression_anderson('sgd')
 
    def test_rmsprop_anderson(self):
        test_linear_regression_anderson('rmsprop')

    def test_adam_anderson(self):
        test_linear_regression_anderson('adam')

###############################################################################


if __name__ == "__main__":
    unittest.main()
