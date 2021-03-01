import torch

import sys
sys.path.append("../src")
sys.path.append("../utils")
import accelerate as accelerate


class FixedPointIteration(object):
    def __init__(self, training_dataloader: torch.utils.data.dataloader.DataLoader, validation_dataloader: torch.utils.data.dataloader.DataLoader,
        learning_rate: float, weight_decay: float = 0.0, verbose: bool = False):
        """

        :type training_dataloader: torch.utils.data.dataloader.DataLoader
        :type validation_dataloader: torch.utils.data.dataloader.DataLoader
        :type learning_rate: float
        :type weight_decay: float
        :type verbose: bool
        """
        self.iteration_counter = 0

        assert isinstance(training_dataloader, torch.utils.data.dataloader.DataLoader)
        assert isinstance(validation_dataloader, torch.utils.data.dataloader.DataLoader)
        self.training_dataloader = training_dataloader
        self.validation_dataloader = validation_dataloader

        assert isinstance(learning_rate, float)
        self.lr = learning_rate

        assert isinstance(weight_decay, float)
        self.weight_decay = weight_decay

        self.model_imported = False
        self.model = None

        self.training_loss_history = []
        self.validation_loss_history = []
        self.criterion_specified = False
        self.criterion = None
        self.optimizer_str = None
        self.optimizer_specified = False
        self.optimizer = None
        self.loss_name = None

        self.verbose = verbose

    def set_zero_grad(self):
        assert self.model_imported
        torch.autograd.zero_grad(self.model.parameters())

    def import_model(self, model):
        assert not self.model_imported
        assert isinstance(model, object)
        self.model = model
        self.model_imported = True

    def get_model(self):
        assert self.model_imported
        return self.model

    def accelerate(self):
        pass

    def train(self, num_epochs, threshold, batch_size):

        assert self.model_imported

        assert self.optimizer_specified
        epoch_counter = 0
        value_loss = float('Inf')

        self.training_loss_history = []
        self.validation_loss_history = []
        self.validation_accuracy = []

        while epoch_counter < num_epochs and value_loss > threshold:

            self.model.get_model().train(True)

            # Training
            for batch_idx, (data, target) in enumerate(self.training_dataloader):
                data, target = (data.to(self.model.get_device()),target.to(self.model.get_device()))
                self.optimizer.zero_grad()
                output = self.model.forward(data)
                loss = self.criterion(output, target)
                loss.backward()
                if self.optimizer_str == 'lbfgs':
                    def closure():
                        if torch.is_grad_enabled():
                            self.optimizer.zero_grad()
                        output = self.model.forward(data)
                        loss = self.criterion(output, target)
                        if loss.requires_grad:
                            loss.backward()
                        return loss
                    self.optimizer.step(closure)
                else:
                    self.optimizer.step()

                self.print_verbose(
                    'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch_counter, batch_idx * len(data), len(self.training_dataloader.dataset), 100.0 * batch_idx / len(self.training_dataloader),loss.item())
                )

            train_loss = loss.item()
            self.training_loss_history.append(train_loss)
            self.accelerate()

            # Validation
            with torch.no_grad():
                self.model.get_model().train(False)
                val_loss = 0.0
                count_val = 0
                correct = 0

                for batch_idx, (data, target) in enumerate(self.validation_dataloader):
                    count_val = count_val + 1
                    data, target = (data.to(self.model.get_device()),target.to(self.model.get_device()))
                    output = self.model.forward(data)
                    loss = self.criterion(output, target)
                    val_loss = val_loss + loss
                    
                    if self.loss_name == 'nll':                    
                        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                        correct += pred.eq(target.view_as(pred)).sum().item()
                    

                val_loss = val_loss / count_val

                self.validation_loss_history.append(val_loss)

                if self.loss_name == 'nll':
                    self.print_verbose(
                        '\n Epoch: '
                        + str(epoch_counter)
                        + ' - Training Loss: '
                        + str(train_loss)
                        + ' - Validation - Loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                            val_loss,
                            correct,
                            len(self.validation_dataloader.dataset),
                            100.0 * correct / len(self.validation_dataloader.dataset),
                        )
                    )
                    self.print_verbose("###############################")
                
                value_loss = val_loss
            epoch_counter = epoch_counter + 1

        return self.training_loss_history, self.validation_loss_history, self.validation_accuracy

    def set_loss_function(self, criterion_string):

        if criterion_string.lower() == 'mse':
            self.criterion = torch.nn.MSELoss()
            self.criterion_specified = True
        elif criterion_string.lower() == 'nll':
            self.criterion = torch.nn.functional.nll_loss
            self.criterion_specified = True
        else:
            raise ValueError("Loss function is not recognized: currently only MSE and CE are allowed")
        self.loss_name = criterion_string

    @property
    def is_loss_function_set(self):
        return self.criterion_specified

    def set_optimizer(self, optimizer_string):

        # we will need the parameters of the deep learning model as inout for the torch.optim object
        # so first we need to make sure that we have already imported the neural network
        assert self.model_imported

        if optimizer_string.lower() == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.get_model().parameters(), lr=self.lr,
                                             weight_decay=self.weight_decay)
            self.optimizer_str = optimizer_string.lower()
            self.optimizer_specified = True
        elif optimizer_string.lower() == 'rmsprop':
            self.optimizer = torch.optim.RMSprop(self.model.get_model().parameters(), lr=self.lr, alpha=0.99,
                                                 weight_decay=self.weight_decay)
            self.optimizer_str = optimizer_string.lower()
            self.optimizer_specified = True
        elif optimizer_string.lower() == 'adam':
            self.optimizer = torch.optim.Adam(self.model.get_model().parameters(), lr=self.lr, betas=(0.9, 0.999),
                                              weight_decay=self.weight_decay)
            self.optimizer_str = optimizer_string.lower()
            self.optimizer_specified = True
        elif optimizer_string.lower() == 'lbfgs':
            self.optimizer = torch.optim.LBFGS(self.model.get_model().parameters(), lr=self.lr, history_size=10,
                                              max_iter=20, line_search_fn='strong_wolfe')
            self.optimizer_str = optimizer_string.lower()
            self.optimizer_specified = True
        else:
            raise ValueError("Optimizer is not recognized: currently only SGD, RMSProp and Adam are allowed")

    @property
    def is_optimizer_set(self):
        return self.optimizer_specified

    def print_verbose(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)



class DeterministicAcceleration(FixedPointIteration):
    def __init__(self,training_dataloader: torch.utils.data.dataloader.DataLoader,validation_dataloader: torch.utils.data.dataloader.DataLoader,
        acceleration_type: str = 'anderson',learning_rate: float = 1e-3,relaxation: float = 0.1,weight_decay: float = 0.0,
        wait_iterations: int = 1, history_depth: int = 15, frequency: int = 1, reg_acc: float = 0.0, store_each_nth: int = 1, verbose: bool = False):

        """

        :type training_dataloader: torch.utils.data.dataloader.DataLoader
        :type validation_dataloader: torch.utils.data.dataloader.DataLoader
        :type acceleration_type: string
        :type learning_rate: float
        :type relaxation: float
        :type weight_decay: float
        :type wait_iterations: int
        :type history_depth: int
        :type frequency: int
        :type reg_acc: float
        :type store_each_nth: int
        :type verbose: bool
        """

        super(DeterministicAcceleration, self).__init__(training_dataloader,validation_dataloader,learning_rate,weight_decay,verbose)
        self.acceleration_type = acceleration_type.lower()
        self.wait_iterations = wait_iterations
        self.relaxation = relaxation
        self.store_each_nth = store_each_nth
        self.history_depth = history_depth
        self.frequency = frequency
        self.reg_acc = reg_acc

    def set_optimizer(self, optimizer_string):
        super(DeterministicAcceleration, self).set_optimizer(optimizer_string)
        accelerate.accelerate(self.optimizer, self.acceleration_type, self.relaxation, self.wait_iterations, self.history_depth, self.store_each_nth, self.frequency, self.reg_acc)
            

