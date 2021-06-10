import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters
import SWA as SWA
from collections import deque
from types import MethodType

import AADL.anderson_acceleration as anderson

def accelerated_step(self, closure=None):
    self.orig_step(closure)
    
    # add current parameters to the history
    self.acc_store_counter += 1
    if self.acc_store_counter >= self.acc_store_each_nth:
        self.acc_store_counter = 0  # reset and continue
        for group, group_hist in zip(self.param_groups, self.acc_param_hist):
            group_hist.append(parameters_to_vector(group['params']).detach())

    # perform acceleration
    self.acc_call_counter += 1
    if (self.acc_call_counter > self.acc_wait_iterations) and (self.acc_call_counter % self.acc_frequency == 0):
        self.acc_call_counter = 0
        for group, group_hist in zip(self.param_groups, self.acc_param_hist):
            if len(group_hist)>=3:
                # make matrix of updates from the history list
                X = torch.stack(list(group_hist), dim=1)

                # compute acceleration
                if self.acc_type == 'anderson':
                    acc_param = anderson.anderson_qr_factorization(X, self.acc_relaxation)
                elif self.acc_type == 'anderson_normal_equation':
                    acc_param = anderson.anderson_normal_equation(X, self.acc_relaxation)                    

                # load acceleration back into model and update history
                vector_to_parameters(acc_param, group['params'])
                group_hist.pop()
                group_hist.append(acc_param)
                

def averaged_step(self, closure=None):
    self.orig_step(closure)
    
    for group, group_hist in zip(self.param_groups, self.avg_param_hist):
        group_hist.append(parameters_to_vector(group['params']).detach())    
        
    #perform moving average
    for group, group_hist in zip(self.param_groups, self.avg_param_hist):
        if len(self.avg_param_hist)==3:
            X = torch.stack(list(group_hist), dim=1)
            average = torch.mean(X, dim=1)
            
            # load acceleration back into model and update history
            vector_to_parameters(average, group['params'])

            for i in range(0,3):
                self.avg_param_hist.pop()
            

def averaged_accelerated_step(self, closure=None):
    self.orig_step(closure)
    
    for group, group_hist in zip(self.param_groups, self.avg_param_hist):
        group_hist.append(parameters_to_vector(group['params']).detach())    
        
    #perform moving average
    for avg_group_hist, acc_group_hist in zip(self.avg_param_hist, self.acc_param_hist):
        if len(avg_group_hist)==3:
            X = torch.stack(list(avg_group_hist), dim=1)
            average = torch.mean(X, dim=1)
            acc_group_hist.append(average)
            
            for i in range(0,3):
                avg_group_hist.pop()
            
    # perform acceleration
    self.acc_call_counter += 1
    if (self.acc_call_counter > self.acc_wait_iterations) and (self.acc_call_counter % self.acc_frequency == 0):
        self.acc_call_counter = 0
        for group, group_hist in zip(self.param_groups, self.acc_param_hist):
            if len(group_hist)>=3:
                # make matrix of updates from the history list
                X = torch.stack(list(group_hist), dim=1)

                # compute acceleration
                if self.acc_type == 'anderson':
                    acc_param = anderson.anderson_qr_factorization(X, self.acc_relaxation)
                elif self.acc_type == 'anderson_normal_equation':
                    acc_param = anderson.anderson_normal_equation(X, self.acc_relaxation, self.acc_reg)                    

                # load acceleration back into model and update history
                vector_to_parameters(acc_param, group['params'])
                group_hist.pop()
                group_hist.append(acc_param)                
                


def accelerate(optimizer, acceleration_type: str = "identity", relaxation: float = 0.1, wait_iterations: int = 1, history_depth: int = 15, store_each_nth: int = 1, frequency: int = 1, reg_acc: float = 0.0, average : bool = False):
    # acceleration options
    optimizer.acc_type            = acceleration_type.lower()
    optimizer.acc_wait_iterations = wait_iterations
    optimizer.acc_relaxation      = relaxation
    optimizer.acc_history_depth   = history_depth
    optimizer.acc_store_each_nth  = store_each_nth
    optimizer.acc_frequency       = frequency
    optimizer.acc_reg             = reg_acc

    # acceleration history
    optimizer.acc_param_hist = [deque([], maxlen=history_depth) for _ in optimizer.param_groups]
    optimizer.avg_param_hist = [deque([], maxlen=history_depth) for _ in optimizer.param_groups]    

    optimizer.acc_call_counter  = 0
    optimizer.acc_store_counter = 0
    
    # redefine step of the optimizer
    optimizer.orig_step = optimizer.step
    
    if average and acceleration_type!="identity":
       optimizer.step      = MethodType(averaged_accelerated_step, optimizer)
    elif  not(average) and acceleration_type!="identity":
       optimizer.step      = MethodType(accelerated_step, optimizer)
    elif average and acceleration_type=="identity":
       optimizer.step      = MethodType(averaged_step, optimizer)

    return optimizer


def remove_acceleration(optimizer):
    if not hasattr(optimizer, 'acc_type'):
        return

    optimizer.step = optimizer.orig_step

    delattr(optimizer, 'acc_type')
    delattr(optimizer, 'acc_wait_iterations')
    delattr(optimizer, 'acc_relaxation')
    delattr(optimizer, 'acc_history_depth')
    delattr(optimizer, 'acc_frequency')
    delattr(optimizer, 'acc_store_each_nth')
    delattr(optimizer, 'acc_reg')
    delattr(optimizer, 'acc_param_hist')
    delattr(optimizer, 'acc_call_counter')
    delattr(optimizer, 'acc_store_counter')
    delattr(optimizer, 'orig_step')

    return optimizer
