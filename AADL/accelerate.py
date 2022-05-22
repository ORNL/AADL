import os

import torch

from AADL.utils import parameters_to_vector_device, vector_to_parameters

from collections import deque
from types import MethodType

import AADL.anderson_acceleration as anderson


_debug = True
_world_size = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ.keys() else 1
_local_rank = int(os.environ["LOCAL_RANK"]) if "LOCAL_RANK" in os.environ.keys() else 0


@torch.no_grad()
def accelerated_step(self, closure=None):
    self.orig_step(closure)

    # add current parameters to the history
    self.acc_store_counter += 1
    if self.acc_store_counter >= self.acc_store_each_nth:
        self.acc_store_counter = 0  # reset and continue
        for group, group_hist in zip(self.param_groups, self.acc_param_hist):
            group_hist.append(parameters_to_vector_device(group['params'], self.history_device))

    # perform acceleration
    self.acc_call_counter += 1
    if (self.acc_call_counter > self.acc_wait_iterations) and (self.acc_call_counter % self.acc_frequency == 0):
        for group, group_hist in zip(self.param_groups, self.acc_param_hist):
            if len(group_hist)>=3:
                # make matrix of updates from the history list
                X = torch.stack(list(group_hist), dim=1).to(device=self.compute_device)

                # compute acceleration
                if self.acc_type == 'anderson':
                    acc_param = anderson.anderson_qr_factorization(X, self.acc_relaxation, self.acc_reg)
                elif self.acc_type == 'anderson_normal_equation':
                    acc_param = anderson.anderson_normal_equation(X, self.acc_relaxation, self.acc_reg)

                # loss after non-accelerated optimizer
                if closure is not None:
                    orig_loss = closure()
                # load acceleration back into model and update history
                vector_to_parameters(acc_param, group['params'])
                if closure is not None:
                    # loss after accelerated optimizer
                    acc_loss = closure()
                    # safeguarding
                    if acc_loss < orig_loss:
                        group_hist.pop()
                        group_hist.append(acc_param.detach().to(device=self.history_device, memory_format=torch.contiguous_format))
                    else:
                        # revert to non-accelerated params
                        vector_to_parameters(group_hist[-1], group['params'])


@torch.no_grad()
def distributed_accelerated_step(self, closure=None):
    acc_loss = orig_loss = self.orig_step(closure)

    # add current parameters to the history
    self.acc_store_counter += 1
    if self.acc_store_counter >= self.acc_store_each_nth:
        self.acc_store_counter = 0  # reset and continue
        for group, group_hist in zip(self.param_groups, self.acc_param_hist):
            group_hist.append(parameters_to_vector_device(group['params'], self.history_device))

    # perform acceleration
    self.acc_call_counter += 1
    if (self.acc_call_counter > self.acc_wait_iterations) and (self.acc_call_counter % self.acc_frequency == 0):
        for group, group_hist in zip(self.param_groups, self.acc_param_hist):
            if len(group_hist)>=3:
                # make matrix of updates from the history list
                X = torch.stack(list(group_hist), dim=1).to(device=self.compute_device)

                # compute acceleration
                if self.acc_type == 'anderson':
                    acc_param = anderson.anderson_qr_factorization(X, self.acc_relaxation, self.acc_reg)
                elif self.acc_type == 'anderson_normal_equation':
                    acc_param = anderson.anderson_normal_equation(X, self.acc_relaxation, self.acc_reg)

                # sync accelerated params across the nodes
                if _world_size>1:
                    self.acc_sync_counter += 1
                    if self.acc_sync_counter % self.acc_sync_frequency == 0:
                        # if _local_rank==0: print(self.acc_sync_counter)
                        self.acc_sync_counter = 0
                        torch.distributed.all_reduce(acc_param, op=torch.distributed.ReduceOp.SUM, group=None, async_op=False)
                        acc_param = acc_param / _world_size

                # load acceleration back into model and update history
                vector_to_parameters(acc_param, group['params'])
                if closure is not None:
                    # loss after accelerated optimizer
                    acc_loss = closure()
                    acc_vote = (acc_loss < orig_loss).float()
                    if _world_size>1:
                        torch.distributed.all_reduce(acc_vote, op=torch.distributed.ReduceOp.SUM, group=None, async_op=False)
                        acc_vote = acc_vote / _world_size
                    # safeguarding
                    if acc_vote>0.9:
                        group_hist.pop()
                        group_hist.append(acc_param.detach().to(device=self.history_device, memory_format=torch.contiguous_format))
                    else:
                        # revert to non-accelerated params
                        vector_to_parameters(group_hist[-1], group['params'])
                        acc_loss = orig_loss

                if _debug:
                    history_list   = [torch.zeros_like(group_hist[-1]) for i in range(_world_size)] if _local_rank==0 else None
                    acc_param_list = [torch.zeros_like(acc_param)      for i in range(_world_size)] if _local_rank==0 else None
                    torch.distributed.gather(group_hist[-1], gather_list=history_list,   dst=0)
                    torch.distributed.gather(acc_param,      gather_list=acc_param_list, dst=0)
                    if _local_rank==0:
                        diff_history_list = 0
                        diff_param_list   = 0
                        for i in range(len(acc_param_list)):
                            diff_history_list = diff_history_list + history_list[i]   - history_list[0]
                            diff_param_list   = diff_param_list   + acc_param_list[i] - acc_param_list[0]
                        print(f'rel_history diff: {torch.max(diff_history_list.abs()).item()/torch.max(history_list[0].abs()).item():.2e}, rel_acc_diff: {torch.max(diff_param_list.abs()).item()/torch.max(acc_param_list[0].abs()).item():.2e},', f'acc_vote: {acc_vote.item():.2f}' if closure is not None else 1)
                        # print(f'rel_history diff: {torch.max(diff_history_list.abs()).item():.2e}, rel_acc_diff: {torch.max(diff_param_list.abs()).item():.2e},', f'acc_vote: {acc_vote.item():.2e}' if closure is not None else 1)
    # torch.distributed.barrier()
    return acc_loss


def averaged_step(self, closure=None):
    self.orig_step(closure)
    
    for group, group_hist in zip(self.param_groups, self.avg_param_hist):
        group_hist.append(parameters_to_vector_device(group['params'], self.history_device))
        
    #perform moving average
    for group, group_hist in zip(self.param_groups, self.avg_param_hist):
        X = torch.stack(list(group_hist), dim=1).to(device=self.compute_device)
        average = torch.mean(X, dim=1)
        std = torch.std(X, dim=1)
            
        if torch.max(std)/torch.max(average)>0.1:
            # load acceleration back into model and update history
            vector_to_parameters(average, group['params'])


def averaged_accelerated_step(self, closure=None):
    self.orig_step(closure)
    
    for group, group_hist in zip(self.param_groups, self.avg_param_hist):
        group_hist.append(parameters_to_vector_device(group['params'], self.history_device))
         
    #perform moving average
    for group, group_hist in zip(self.param_groups, self.avg_param_hist):
        X = torch.stack(list(group_hist), dim=1).to(device=self.compute_device)
        average = torch.mean(X, dim=1)
        std = torch.std(X, dim=1)
            
        #print(torch.norm(std), torch.norm(average), torch.norm(std)/torch.norm(average))
            
        if torch.max(std)/torch.max(average)>0.1:
            # load acceleration back into model and update history
            vector_to_parameters(average, group['params'])

    # add current parameters to the history
    self.acc_store_counter += 1
    if self.acc_store_counter >= self.acc_store_each_nth:
        self.acc_store_counter = 0  # reset and continue
        for group, group_hist in zip(self.param_groups, self.acc_param_hist):
            group_hist.append(parameters_to_vector_device(group['params'], self.history_device))
            
    # perform acceleration
    self.acc_call_counter += 1
    if (self.acc_call_counter > self.acc_wait_iterations) and (self.acc_call_counter % self.acc_frequency == 0):
        for group, group_hist in zip(self.param_groups, self.acc_param_hist):
            if len(group_hist)>=3:
                # make matrix of updates from the history list
                X = torch.stack(list(group_hist), dim=1).to(device=self.compute_device)

                # compute acceleration
                if self.acc_type == 'anderson':
                    acc_param = anderson.anderson_qr_factorization(X, self.acc_relaxation, self.acc_reg) 
                elif self.acc_type == 'anderson_normal_equation':
                    acc_param = anderson.anderson_normal_equation(X, self.acc_relaxation, self.acc_reg)                    

                # loss after non-accelerated optimizer
                if closure is not None:
                    orig_loss = closure()
                # load acceleration back into model and update history
                vector_to_parameters(acc_param, group['params'])

                if closure is not None:
                    # loss after accelerated optimizer
                    acc_loss = closure()
                    # safeguarding
                    if acc_loss < orig_loss:
                        group_hist.pop()
                        group_hist.append(acc_param.detach().to(device=self.history_device, memory_format=torch.contiguous_format))
                    else:
                        # revert to non-accelerated params
                        vector_to_parameters(group_hist[-1], group['params'])


def accelerate(optimizer, acceleration_type: str = "identity", relaxation: float = 0.1, wait_iterations: int = 1, history_depth: int = 15, store_each_nth: int = 1, frequency: int = 1, reg_acc: float = 0.0, average : bool = False, history_device: str = "cpu", compute_device: str = "cpu", distributed: bool = False, sync_frequency: int = 1):
    # acceleration options
    optimizer.acc_type            = acceleration_type.lower()
    optimizer.acc_wait_iterations = wait_iterations
    optimizer.acc_relaxation      = relaxation
    optimizer.acc_history_depth   = history_depth
    optimizer.acc_store_each_nth  = store_each_nth
    optimizer.acc_frequency       = frequency
    optimizer.acc_sync_frequency  = sync_frequency
    optimizer.acc_reg             = reg_acc

    # acceleration history
    optimizer.acc_param_hist = [deque([], maxlen=history_depth) for _ in optimizer.param_groups]
    optimizer.avg_param_hist = [deque([], maxlen=history_depth) for _ in optimizer.param_groups]    

    optimizer.acc_call_counter  = 0
    optimizer.acc_store_counter = 0
    optimizer.acc_sync_counter  = 0
    
    # redefine step of the optimizer
    optimizer.orig_step = optimizer.step
    
    if average and acceleration_type!="identity":
       optimizer.step      = MethodType(averaged_accelerated_step, optimizer)
    elif  not(average) and acceleration_type!="identity":
       optimizer.step = MethodType(distributed_accelerated_step, optimizer) if distributed else MethodType(accelerated_step, optimizer)
    elif average and acceleration_type=="identity":
       optimizer.step      = MethodType(averaged_step, optimizer)

    optimizer.history_device = history_device
    optimizer.compute_device = compute_device

    return optimizer


def distributed_accelerate(optimizer, **kwargs):
    return accelerate(optimizer, **kwargs, distributed=True)


def remove_acceleration(optimizer):
    if not hasattr(optimizer, 'acc_type'):
        return

    optimizer.step = optimizer.orig_step

    delattr(optimizer, 'acc_type')
    delattr(optimizer, 'acc_wait_iterations')
    delattr(optimizer, 'acc_relaxation')
    delattr(optimizer, 'acc_history_depth')
    delattr(optimizer, 'acc_frequency')
    delattr(optimizer, 'acc_sync_frequency')
    delattr(optimizer, 'acc_store_each_nth')
    delattr(optimizer, 'acc_reg')
    delattr(optimizer, 'acc_param_hist')
    delattr(optimizer, 'acc_call_counter')
    delattr(optimizer, 'acc_store_counter')
    delattr(optimizer, 'acc_sync_counter')
    delattr(optimizer, 'orig_step')
    delattr(optimizer, 'history_device')
    delattr(optimizer, 'compute_device')

    return optimizer
