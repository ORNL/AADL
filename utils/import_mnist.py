import pickle
import numpy
import torch
import pandas


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return numpy.eye(num_classes, dtype='uint8')[y]

def import_mnist_data():
    df = pandas.read_csv('./datasets/mnist/X_mnist.csv', header=None)
    x_sample = df.values
    df = pandas.read_csv('./datasets/mnist/y_mnist.csv', header=None)
    y_vals = df.values
    x_sample = x_sample.reshape(x_sample.shape[0], 1, int(numpy.sqrt(x_sample.shape[1])), int(numpy.sqrt(x_sample.shape[1])))
    y_vals = y_vals.reshape(y_vals.shape[0], 1)
    y_vals = y_vals.astype(int)

    x_sample = x_sample.astype("float")
    x_sample /= 255.0

    # Extract important features of the data
    input_dim = x_sample.shape[1:]
    output_dim = int(10)

    x_train = x_sample[0:60000, :, :]
    x_test = x_sample[60000:, :, :]
    y_train = y_vals[0:60000, :]
    y_test = y_vals[60000:, :]

    # Doubles must be converted to Floats before passing them to a neural network model
    x_train = torch.from_numpy(x_train).float()
    x_test = torch.from_numpy(x_test).float()
    y_train = torch.from_numpy(y_train).long()
    y_train = torch.squeeze(y_train,1)
    y_test = torch.from_numpy(y_test).long()
    y_test = torch.squeeze(y_test,1)

    return input_dim, output_dim, x_train, x_test, y_train, y_test
