#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 13 17:17:05 2021

@author: 7ml
"""
import sys
import torch
import numpy
import unittest
from torch.utils.data import Dataset

sys.path.append('../utils')
from optimizers import FixedPointIteration, DeterministicAcceleration
sys.path.append('../model_zoo')
from LinearRegression_models import LinearRegression


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


def linear_regression(slope, intercept, num_points, optimizer_str):
    input_dim, output_dim, dataset = linear_data(slope, intercept, num_points)
    use_bias = True
    learning_rate = 1e-5
    weight_decay = 0.0
    batch_size = 1
    epochs = 100
    threshold = 1e-8

    training_dataloader = torch.utils.data.DataLoader(dataset, batch_size)
    validation_dataloader = torch.utils.data.DataLoader(dataset, batch_size)

    model = LinearRegression(input_dim, output_dim, use_bias)
    optimizer_classic = FixedPointIteration(training_dataloader, validation_dataloader, learning_rate, weight_decay)
    optimizer_classic.import_model(model)
    optimizer_classic.set_loss_function('nonconvex')
    optimizer_classic.set_optimizer(optimizer_str)
    training_classic_loss_history, validation_classic_loss_history, _ = optimizer_classic.train(epochs, threshold, batch_size)

    weights = list(model.get_model().parameters())

    return weights[0].item(), weights[1].item(), validation_classic_loss_history

def linear_regression_average(slope, intercept, num_points, optimizer_str):
    input_dim, output_dim, dataset = linear_data(slope, intercept, num_points)
    acceleration_type = "identity"
    use_bias = True
    learning_rate = 1e-5
    relaxation = 1.0
    weight_decay = 0.0
    batch_size = 1
    epochs = 100
    threshold = 1e-8
    wait_iterations = 1
    history_depth = 3
    frequency = 3
    reg_acc = 1e-8
    store_each_nth = frequency
    average = True
    safeguard = True

    training_dataloader = torch.utils.data.DataLoader(dataset, batch_size)
    validation_dataloader = torch.utils.data.DataLoader(dataset, batch_size)

    model = LinearRegression(input_dim, output_dim, use_bias)
    optimizer_anderson = DeterministicAcceleration(training_dataloader,validation_dataloader,acceleration_type,learning_rate,relaxation,weight_decay,wait_iterations,history_depth,
        frequency,reg_acc,store_each_nth, average, safeguard)
    optimizer_anderson.import_model(model)
    optimizer_anderson.set_loss_function('nonconvex')
    optimizer_anderson.set_optimizer(optimizer_str)
    training_anderson_loss_history, validation_anderson_loss_history, _ = optimizer_anderson.train(epochs, threshold, batch_size)

    weights = list(model.get_model().parameters())

    return weights[0].item(), weights[1].item(), validation_anderson_loss_history


def linear_regression_anderson(slope, intercept, num_points, optimizer_str):
    input_dim, output_dim, dataset = linear_data(slope, intercept, num_points)
    acceleration_type = "anderson"
    use_bias = True
    learning_rate = 1e-5
    relaxation = 1.0
    weight_decay = 0.0
    batch_size = 1
    epochs = 100
    threshold = 1e-8
    wait_iterations = 1
    history_depth = 5
    frequency = 3
    reg_acc = 1e-8
    store_each_nth = frequency
    average = True
    safeguard = True

    training_dataloader = torch.utils.data.DataLoader(dataset, batch_size)
    validation_dataloader = torch.utils.data.DataLoader(dataset, batch_size)

    model = LinearRegression(input_dim, output_dim, use_bias)
    optimizer_anderson = DeterministicAcceleration(training_dataloader,validation_dataloader,acceleration_type,learning_rate,relaxation,weight_decay,wait_iterations,history_depth,
        frequency,reg_acc,store_each_nth, average, safeguard)
    
    optimizer_anderson.import_model(model)
    optimizer_anderson.set_loss_function('nonconvex')
    optimizer_anderson.set_optimizer(optimizer_str)
    training_anderson_loss_history, validation_anderson_loss_history, _ = optimizer_anderson.train(epochs, threshold, batch_size)

    weights = list(model.get_model().parameters())

    return weights[0].item(), weights[1].item(), validation_anderson_loss_history


def test_linear_regression(optimiser):
    num_points = 3000
    straight_line_parameters = torch.ones(2, 1)
    slope = straight_line_parameters[0].item()
    intercept = straight_line_parameters[1].item()
    numeric_slope, numeric_intercept, history = linear_regression(slope, intercept, num_points, optimiser)
    print(optimiser+" converged in "+ str(len(history))+" iterations "+"\n exact slope: "+ str(slope)+"  - "+ " numerical slope: "+str(numeric_slope)+"\n"+" exact intercept: "
        + str(intercept)+" - "+" numerical intercept: "+str(numeric_intercept))
    assert(abs((slope - numeric_slope)) < 1e-3 and abs((intercept - numeric_intercept)) < 1e-3)
    
def test_linear_regression_average(optimiser):
    num_points = 3000
    straight_line_parameters = torch.ones(2, 1)
    slope = straight_line_parameters[0].item()
    intercept = straight_line_parameters[1].item()
    numeric_slope, numeric_intercept, history = linear_regression_average(
        slope, intercept, num_points, optimiser
    )
    print(optimiser+" + Average converged in "+ str(len(history))+" iterations "+"\n exact slope: "+ str(slope)+"  - "+ " numerical slope: "+str(numeric_slope)+"\n"+" exact intercept: "
        + str(intercept)+" - "+" numerical intercept: "+str(numeric_intercept))
    assert(abs((slope - numeric_slope)) < 1e-3 and abs((intercept - numeric_intercept)) < 1e-3)    


def test_linear_regression_anderson(optimiser):
    num_points = 3000
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

    """
    def test_asgd(self):
        test_linear_regression('asgd')

    def test_rmsprop(self):
        test_linear_regression('rmsprop')
        

    def test_adam(self):
        test_linear_regression('adam')
    """
    def test_sgd_average(self):
        test_linear_regression_average('sgd')

    def test_sgd_anderson(self):
        test_linear_regression_anderson('sgd')
        
    """

    def test_asgd_anderson(self):
        test_linear_regression_anderson('asgd')

    def test_rmsprop_anderson(self):
        test_linear_regression_anderson('rmsprop')
    

    def test_adam_anderson(self):
        test_linear_regression_anderson('adam')
    """
    

###############################################################################


if __name__ == "__main__":
    unittest.main()
