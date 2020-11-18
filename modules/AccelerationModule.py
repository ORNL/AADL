import torch
import numpy as np
import sys

sys.path.append("../utils")
import rna_acceleration as rna


class AccelerationModule(object):
    def __init__(self, model: torch.nn.Module, window_depth: int = 15, reg_acc: float = 1e-5, store_each: int = 1):
        """

        :param model: :type torch.nn.Module
        :param window_depth: :type int
        :param reg_acc: :type float
        :param store_each: :type int
        """
        self.store_counter = 0

        self.x_hist = []
        self.window_depth = window_depth
        self.reg_acc = reg_acc
        self.input_shape = dict()
        self.store_each = store_each

        key = 0
        for param in model.parameters():
            data_np = param.data.cpu().numpy()
            self.input_shape[key] = (data_np.shape, data_np.size)
            key += 1

    def extract_x(self, model):
        new_x = []

        for param in model.parameters():
            param_np = param.data.cpu().numpy().ravel()
            new_x.append(param_np)

        new_x_cat = np.array(np.concatenate(new_x))

        return new_x_cat

    def store(self, model):
        self.store_counter += 1
        if self.store_counter >= self.store_each:
            self.store_counter = 0  # reset and continue
        else:
            return  # don't store

        if len(
                self.x_hist) > self.window_depth:  # with this, len(x_hist) < window_depth+1, so number of coeffs < window_depth
            self.x_hist.pop(0)

        self.x_hist.append(self.extract_x(model))

    def load_param_in_model(self, x, model, x0=None, step_size=1):
        first_idx = 0
        last_idx = 0

        key = 0
        for param in model.parameters():
            (shape, num_elem) = self.input_shape[key]
            last_idx = first_idx + num_elem
            param.data = torch.FloatTensor(
                x[first_idx:last_idx].reshape(shape)
                if x0 is None else
                (1 - step_size) * x0[first_idx:last_idx].reshape(shape) + step_size * x[first_idx:last_idx].reshape(
                    shape)
            )

            first_idx = last_idx
            key += 1

    def min_eigenval(self):
        x_hist_np = np.array(self.x_hist).transpose()
        return rna.min_eignevalRR(x_hist_np)

    def accelerate(self, model, step_size: float = 1):

        if len(self.x_hist) < 3:  # Cannot accelerate when number of points is below 3
            self.load_param_in_model(np.array(self.x_hist[-1]), model)
            return 1

        x_hist_np = np.array(self.x_hist).transpose()
        x_acc, c = rna.rna(x_hist_np, self.reg_acc)

        if step_size == 1.0:
            self.load_param_in_model(x_acc, model)
        else:
            self.load_param_in_model(x_acc, model, self.x_hist[-1], step_size)

        return c
