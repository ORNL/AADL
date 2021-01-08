import torch
import numpy as np
import sys

sys.path.append("../utils")
import rna_acceleration as rna
import anderson_acceleration as anderson


class AccelerationModule(object):
    def __init__(self, acceleration_type: str, model: torch.nn.Module, history_depth: int = 15, reg_acc: float = 1e-5, store_each_nth: int = 1):
        """

        :param model: :type torch.nn.Module
        :param history_depth: :type int
        :param reg_acc: :type float
        :param store_each_nth: :type int
        """
        self.acceleration_type = acceleration_type.lower()
        self.store_counter = 0

        self.x_hist = []
        self.history_depth = history_depth
        self.reg_acc = reg_acc
        self.input_shape = dict()
        self.store_each_nth = store_each_nth

        key = 0
        for param in model.parameters():
            self.input_shape[key] = (param.data.shape, param.data.numel())
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
        if self.store_counter >= self.store_each_nth:
            self.store_counter = 0  # reset and continue
        else:
            return  # don't store

        if len(
                self.x_hist) > self.history_depth:  # with this, len(x_hist) < history_depth+1, so number of coeffs < history_depth
            self.x_hist.pop(0)

        self.x_hist.append(self.extract_x(model))

    def load_param_in_model(self, x, model, x0=None, step_size=1):
        first_idx = 0
        last_idx = 0

        key = 0
        for param in model.parameters():
            (shape, num_elem) = self.input_shape[key]
            last_idx = first_idx + num_elem
            if self.acceleration_type == 'rna':
                param.data = torch.tensor(
                    x[first_idx:last_idx].reshape(shape)
                    if x0 is None else
                    (1 - step_size) * x0[first_idx:last_idx].reshape(shape) + step_size * x[first_idx:last_idx].reshape(
                        shape), dtype=torch.float
                )
            elif self.acceleration_type == 'anderson':
                param.data = torch.tensor(
                    x[first_idx:last_idx].reshape(shape)
                    if x0 is None else
                    x0[first_idx:last_idx].reshape(shape) + step_size * x[first_idx:last_idx].reshape(
                        shape), dtype=torch.float
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
        
        if self.acceleration_type == 'anderson':
            x_acc, c = anderson.anderson(x_hist_np, self.reg_acc)
            
        elif self.acceleration_type == 'rna':
            x_acc, c = rna.rna(x_hist_np, self.reg_acc)
    
        if step_size == 1.0:
            self.load_param_in_model(x_acc, model)
        else:
            self.load_param_in_model(x_acc, model, self.x_hist[-1], step_size)

        return c
