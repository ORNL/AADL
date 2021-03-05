#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@authors: Massimiliano Lupo Pasini (e-mail: lupopasinim@ornl.gov.gov)
        : Miroslav Stoyanov (e-mail: stoyanovmk@ornl.gov.gov)
        : Viktor Reshniak (e-mail: reshniakv@ornl.gov.gov)

Usage:
  main.py (-h | --help)
  main.py [-c CONFIG_FILE] [--number_runs] [--verbose] [--display] [--dataset] [--subsample] [--classification] [--model] [--neurons] [--layers] [-a ACTIVATION] [-b BIAS]
          [--optimizer] [-e EPOCHS] [-l LEARNING_RATE] [--threshold] [--batch] [-p PENALIZATION] [--acceleration] [-d DEPTH] [-w WAIT_ITERATIONS] [-f FREQUENCY]
          [-s STORE_EACH] [-r REGULARIZATION] [--relaxation]

Options:
  -h, --help                  Show this screen
  --number_runs               Number of runs, each one using a different fixed random seed
  --version                   Show version
  --verbose                   Level of verbosity
  --display                   Use matplotlib to plot results
  -c, --config=<str>          Filename containing configuration parameters
  --dataset                   Dataset used for training. GRADUATE_ADMISSION, MNIST, CIFAR10 [default: MNIST]
  --subsample                 Number going 0 through 1 for the percentage of the original data used [default: 1.0]
  --classification            Type of problem: classification or regression
  --model                     Implementation of NN model. Multi-layer perceptrons NN (MLP), convolutional NN (CNN)
  --neurons                   Number of neurons per layer
  --layers                    Number of hidden layers
  -a, --activation=<str>      Type of activation function [default: RELU]
  -b, --bias                  Use bias in the regression model [default: True]
  --optimizer                 Optimizer name [default: SGD]
  -e, --epochs=<n>            Number of epochs [default: 1]
  -l, --learning_rate=<f>     Learning rate [default: 0.01]
  --threshold                 Stopping criterion for the training [default: 1e-4]
  --batch                     Size of the batch for the optimizer [default: 1]
  --acceleration              Type of accelerarion performed. ANDERSON, RNA [default: ANDERSON]
  -p, --penalization=<f>      Weight decay for the L2 penalization during the training of the neural network [default: 0.0]
  -d, --history_depth=<m>     Depth of window history for anderson [default: 5]
  -w, --wait_iterations=<n>   Wait an initial number of classic optimizer iterations before starting with anderson [default: 1]
  -f, --frequency=<n>         Number of epochs performed between two consecutive anderson accelerations [default: 1]
  -s, --store_each_nth=<n>    Number of epochs performed between two consecutive storages of the iterations in the columns of matrix R to perform least squares [default: 1]
  -r, --regularization=<f>    Regularization parameter for L2 penalization for the least squares problem solved to perform the acceleration [default: 0.0]
  --relaxation                relaxation parameter to mix the past iterate with the acceleration update and generate the new iterate [default: 0.1]
"""

from docopt import docopt
import yaml
import sys
from copy import deepcopy

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import torch
from numpy.core.defchararray import lower

import numpy

sys.path.append("../../utils")
from optimizers import FixedPointIteration, DeterministicAcceleration
sys.path.append("../../model_zoo")
from NN_models import MLP
from graduate_admission_dataloader import dataloader

from gpu_detection import get_gpu

plt.rcParams.update({'font.size': 16})

def merge_args(cmdline_args, config_args):
    for key in config_args.keys():
        if key not in cmdline_args:
            sys.exit(
                'Error: unknown key in the configuration file \"{}\"'.format(
                    key
                )
            )

    args = {}
    args.update(cmdline_args)
    args.update(config_args)

    return args


def get_options():
    args = docopt(__doc__, version='Accelerated Training 0.0')

    # strip -- from names
    args = {key[2:]: value for key, value in args.items()}

    config_args = {}
    if args['config']:
        with open(args['config']) as f:
            config_args = yaml.load(f, Loader=yaml.FullLoader)

    # strip certain options
    # this serves 2 purposes:
    # - This option have already been parsed, and have no meaning as input
    #   parameters
    # - The config file options would be not allowed to contain them
    for skip_option in {'config', 'help'}:
        del args[skip_option]

    return merge_args(args, config_args)


if __name__ == '__main__':
    config = get_options()

    number_runs = int(config['number_runs'])
    verbose = bool(config['verbose'])
    classification_problem = bool(config['classification'])

    # Specify name of the dataset and percentage of the entire data volume to sample
    dataset_name = lower(config['dataset'])
    subsample_factor = float(config['subsample'])

    # Setting for the neural network
    model_name = str(lower(config['model']))
    num_neurons = int(config['neurons'])
    num_layers = int(config['layers'])
    num_neurons_list = [num_neurons for i in range(num_layers)]
    activation = str(lower(config['activation']))
    use_bias = bool(config['bias'])

    # Generic parameters for optimizer
    optimizer_name = str(lower(config['optimizer']))
    epochs = int(config['epochs'])
    learning_rate = float(config['learning_rate'])
    threshold = float(config['threshold'])
    batch_size = int(config['batch'])
    weight_decay = float(config['penalization'])

    # Parameters for acceleration
    acceleration = str(lower(config['acceleration']))
    wait_iterations = int(config['wait_iterations'])
    history_depth = int(config['history_depth'])
    frequency = int(config['frequency'])
    store_each_nth = int(config['store_each_nth'])
    reg_acc = float(config['regularization'])
    relaxation = float(config['relaxation'])

    # The only reason why I do this workaround (not necessary now) is because
    # I am thinking to the situation where one MPI process has multiple gpus available
    # In that case, the argument passed to get_gpu may be a numberID > 0
    available_device = get_gpu(0)

    # Generate dataloaders for training and validation
    (
        input_dim,
        output_dim,
        training_dataloader,
        validation_dataloader,
    ) = dataloader(dataset_name, subsample_factor, batch_size)

    color = cm.rainbow(numpy.linspace(0, 1, number_runs))

    for iteration in range(0, number_runs):

        torch.manual_seed(iteration)

        # Define deep learning model
        model_classic = MLP(input_dim,output_dim,num_neurons_list,use_bias,activation,classification_problem,available_device)

        model_anderson = deepcopy(model_classic)

        # For classification problems, the loss function is the negative log-likelihood (nll)
        # For regression problems, the loss function is the mean squared error (mse)
        loss_function_name = 'nll' if classification_problem else 'mse'

        # Define the standard optimizer which is used as point of reference to assess the improvement provided by the
        # acceleration
        optimizer_classic = FixedPointIteration(training_dataloader,validation_dataloader,learning_rate,weight_decay,verbose)

        optimizer_classic.import_model(model_classic)
        optimizer_classic.set_loss_function(loss_function_name)
        optimizer_classic.set_optimizer(optimizer_name)

        (
            training_classic_loss_history,
            validation_classic_loss_history,
            validation_classic_accuracy_history,
        ) = optimizer_classic.train(epochs, threshold, batch_size)

        optimizer_anderson = DeterministicAcceleration(
            training_dataloader,
            validation_dataloader,
            acceleration,
            learning_rate,
            relaxation,
            weight_decay,
            wait_iterations,
            history_depth,
            frequency,
            reg_acc,
            store_each_nth,
            verbose,
        )

        optimizer_anderson.import_model(model_anderson)
        optimizer_anderson.set_loss_function(loss_function_name)
        optimizer_anderson.set_optimizer(optimizer_name)

        (
            training_anderson_loss_history,
            validation_anderson_loss_history,
            validation_anderson_accuracy_history,
        ) = optimizer_anderson.train(epochs, threshold, batch_size)

        if config['display']:
            epochs1 = range(1, len(training_classic_loss_history) + 1)
            epochs2 = range(1, len(training_anderson_loss_history) + 1)
            if len(validation_classic_accuracy_history) > 0:
                plt.plot(epochs1,validation_classic_accuracy_history,color=color[iteration],linestyle='-')
                plt.plot(epochs2,validation_anderson_accuracy_history,color=color[iteration],linestyle='--')
            else:
                plt.plot(epochs1,validation_classic_loss_history,color=color[iteration],linestyle='-')
                plt.plot(epochs2,validation_anderson_loss_history,color=color[iteration],linestyle='--')                
            plt.yscale('log')
            plt.title('Validation loss function')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            #plt.legend()
            plt.draw()
            plt.savefig('validation_loss_plot')

            plt.tight_layout()
