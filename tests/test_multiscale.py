import sys
import torch
import unittest

sys.path.append('../utils')
from optimizers import FixedPointIteration, DeterministicAcceleration
sys.path.append('../model_zoo')
from TestFunctions_models import Paraboloid


###############################################################################


def test_multiscale_paraboloid(dim=100, condition_number=1.e3, optimizer='sgd', lr=1.e-4, w_decay=0.0, epochs=10000, threshold=1.e-8):
    batch_size = dim

    dataset = torch.utils.data.TensorDataset(torch.zeros(dim), torch.zeros(dim))

    training_dataloader = torch.utils.data.DataLoader(dataset, batch_size)
    validation_dataloader = torch.utils.data.DataLoader(dataset, batch_size)

    model = Paraboloid(dim, condition_number=condition_number)
    optimizer_classic = FixedPointIteration(training_dataloader, validation_dataloader, lr, w_decay)
    optimizer_classic.import_model(model)
    optimizer_classic.set_loss_function('mse')
    optimizer_classic.set_optimizer(optimizer)
    _, validation_classic_loss_history, _ = optimizer_classic.train(epochs, threshold, batch_size)

    weights = model.get_model().get_weight()

    return weights.abs().max(), validation_classic_loss_history


def test_multiscale_paraboloid_anderson(dim=100,condition_number=1.0e3,optimizer='sgd',lr=1.0e-4,w_decay=0.0,epochs=10000,threshold=1.0e-8,):
    batch_size = dim
    acceleration_type = "anderson"
    use_bias = True
    relaxation = 0.1
    weight_decay = 0.0
    threshold = 1e-8
    wait_iterations = 1
    history_depth = 10
    frequency = 3
    reg_acc = 1e-7
    store_each_nth = frequency
    average = False
    safeguard = True

    dataset = torch.utils.data.TensorDataset(torch.zeros(dim), torch.zeros(dim))
    training_dataloader = torch.utils.data.DataLoader(dataset, batch_size)
    validation_dataloader = torch.utils.data.DataLoader(dataset, batch_size)

    model = Paraboloid(dim, condition_number=condition_number)
    optimizer_anderson = DeterministicAcceleration(training_dataloader,validation_dataloader,acceleration_type,lr,relaxation,weight_decay,wait_iterations,history_depth,
        frequency,reg_acc,store_each_nth, average, safeguard)

    optimizer_anderson.import_model(model)
    optimizer_anderson.set_loss_function('mse')
    optimizer_anderson.set_optimizer(optimizer)
    _, validation_anderson_loss_history, _ = optimizer_anderson.train(epochs, threshold, batch_size)

    weights = model.get_model().get_weight()

    return weights.abs().max(), validation_anderson_loss_history


###############################################################################


class TestMultiscaleParaboloid(unittest.TestCase):
    

    def test_2d_well_conditioned_paraboloid_sgd(self):
        weight_sum, history = test_multiscale_paraboloid(dim=2, condition_number=1, optimizer='sgd', lr=1.e-2, epochs=10000)
        print("Well conditioned problem - 2d SGD finished after "+str(len(history))+" iterations "+"\n exact weight sum: 0"+"  - "+" numerical weight sum: "+str(weight_sum))
        self.assertTrue(weight_sum<1e-3)

    def test_2d_well_conditioned_paraboloid_rmsprop(self):
        weight_sum, history = test_multiscale_paraboloid(dim=2, condition_number=1, optimizer='rmsprop', lr=1.e-2, epochs=10000)
        print("Well conditioned problem - 2d RMSProp finished after "+str(len(history))+" iterations "+"\n exact weight sum: 0"+"  - "+" numerical weight sum: "+str(weight_sum))
        self.assertTrue(weight_sum<1e-3)

    def test_2d_well_conditioned_paraboloid_adam(self):
        weight_sum, history = test_multiscale_paraboloid(dim=2, condition_number=1, optimizer='adam', lr=1.e-2, epochs=10000)
        print("Well conditioned problem - 2d Adam finished after "+str(len(history))+" iterations "+"\n exact weight sum: 0"+"  - "+" numerical weight sum: "+str(weight_sum))
        self.assertTrue(weight_sum<1e-3)


    ###########


    def test_2d_well_conditioned_paraboloid_sgd_anderson(self):
        weight_sum, history = test_multiscale_paraboloid_anderson(dim=2, condition_number=1, optimizer='sgd', lr=1.e-3, epochs=10000)
        print("Well conditioned problem - 2d SGD + Anderson finished after "+str(len(history))+" iterations "+"\n exact weight sum: 0"+"  - "+" numerical weight sum: "+str(weight_sum))
        self.assertTrue(weight_sum<1e-3)

    def test_2d_well_conditioned_paraboloid_rmsprop_anderson(self):
        weight_sum, history = test_multiscale_paraboloid_anderson(dim=2, condition_number=1, optimizer='rmsprop', lr=1.e-3, epochs=10000)
        print("Well conditioned problem - 2d RMSProp + Anderson finished after "+str(len(history))+" iterations "+"\n exact weight sum: 0"+"  - "+" numerical weight sum: "+str(weight_sum))
        self.assertTrue(weight_sum<1e-3)

    def test_2d_well_conditioned_paraboloid_adam_anderson(self):
        weight_sum, history = test_multiscale_paraboloid_anderson(dim=2, condition_number=1, optimizer='adam', lr=1.e-3, epochs=10000)
        print("Well conditioned problem - 2d Adam + Anderson finished after "+str(len(history))+" iterations "+"\n exact weight sum: 0"+"  - "+" numerical weight sum: "+str(weight_sum))
        self.assertTrue(weight_sum<1e-3)
        
    ###########


    def test_100d_well_conditioned_paraboloid_sgd(self):
        weight_sum, history = test_multiscale_paraboloid_anderson(dim=2, condition_number=1, optimizer='sgd', lr=1.e-3, epochs=1000)
        print("Well conditioned problem - 100d SGD finished after "+str(len(history))+" iterations "+"\n exact weight sum: 0"+"  - "+" numerical weight sum: "+str(weight_sum))
        self.assertTrue(weight_sum<1e-3)


    def test_100d_well_conditioned_paraboloid_rmsprop(self):
        weight_sum, history = test_multiscale_paraboloid_anderson(dim=100, condition_number=1, optimizer='rmsprop', lr=1.e-3, epochs=10000)
        print("Well conditioned problem - 100d RMSProp finished after "+str(len(history))+" iterations "+"\n exact weight sum: 0"+"  - "+" numerical weight sum: "+str(weight_sum))
        self.assertTrue(weight_sum<1e-3)

    def test_100d_well_conditioned_paraboloid_adam(self):
        weight_sum, history = test_multiscale_paraboloid_anderson(dim=100, condition_number=1, optimizer='adam', lr=1.e-3, epochs=10000)
        print("Well conditioned problem - 100d Adam finished after "+str(len(history))+" iterations "+"\n exact weight sum: 0"+"  - "+" numerical weight sum: "+str(weight_sum))
        self.assertTrue(weight_sum<1e-3)


    def test_100d_well_conditioned_paraboloid_sgd_anderson(self):
        weight_sum, history = test_multiscale_paraboloid_anderson(dim=2, condition_number=1, optimizer='sgd', lr=1.e-3, epochs=1000)
        print("Well conditioned problem - 100d SGD + Anderson finished after "+str(len(history))+" iterations "+"\n exact weight sum: 0"+"  - "+" numerical weight sum: "+str(weight_sum))
        self.assertTrue(weight_sum<1e-3)
    
 
    def test_100d_well_conditioned_paraboloid_rmsprop_anderson(self):
        weight_sum, history = test_multiscale_paraboloid_anderson(dim=100, condition_number=1, optimizer='rmsprop', lr=1.e-3, epochs=10000)
        print("Well conditioned problem - 100d RMSProp + Anderson finished after "+str(len(history))+" iterations "+"\n exact weight sum: 0"+"  - "+" numerical weight sum: "+str(weight_sum))
        self.assertTrue(weight_sum<1e-3)

    def test_100d_well_conditioned_paraboloid_adam_anderson(self):
        weight_sum, history = test_multiscale_paraboloid_anderson(dim=100, condition_number=1, optimizer='adam', lr=1.e-3, epochs=10000)
        print("Well conditioned problem - 100d Adam + Anderson finished after "+str(len(history))+" iterations "+"\n exact weight sum: 0"+"  - "+" numerical weight sum: "+str(weight_sum))
        self.assertTrue(weight_sum<1e-3)
        


    ###########


    def test_100d_ill_conditioned_paraboloid_sgd(self):
        weight_sum, history = test_multiscale_paraboloid(dim=100, condition_number=1.e3, optimizer='sgd', lr=1.e-3, epochs=10000)
        print("Ill conditioned problem - 100d SGD finished after "+str(len(history))+" iterations "+"\n exact weight sum: 0"+"  - "+" numerical weight sum: "+str(weight_sum))
        self.assertTrue(weight_sum<1e-3)

    def test_100d_ill_conditioned_paraboloid_rmsprop(self):
        weight_sum, history = test_multiscale_paraboloid(dim=100, condition_number=1.e3, optimizer='rmsprop', lr=1.e-3, epochs=10000)
        print("Ill conditioned problem - 100d RMSProp finished after "+str(len(history))+" iterations "+"\n exact weight sum: 0"+"  - "+" numerical weight sum: "+str(weight_sum))
        self.assertTrue(weight_sum<1e-3)

    def test_100d_ill_conditioned_paraboloid_adam(self):
        weight_sum, history = test_multiscale_paraboloid(dim=100, condition_number=1.e3, optimizer='adam', lr=1.e-3, epochs=10000)
        print("Ill conditioned problem - 100d Adam finished after "+str(len(history))+" iterations "+"\n exact weight sum: 0"+"  - "+" numerical weight sum: "+str(weight_sum))
        self.assertTrue(weight_sum<1e-3)


    ###########


    def test_100d_ill_conditioned_paraboloid_sgd_anderson(self):
        weight_sum, history = test_multiscale_paraboloid_anderson(dim=100, condition_number=1.e3, optimizer='sgd', lr=1.e-3, epochs=10000)
        print("Ill conditioned problem - 100d SGD + Anderson finished after "+str(len(history))+" iterations "+"\n exact weight sum: 0"+"  - "+" numerical weight sum: "+str(weight_sum))
        self.assertTrue(weight_sum<1e-3)

    def test_100d_ill_conditioned_paraboloid_rmsprop_anderson(self):
        weight_sum, history = test_multiscale_paraboloid_anderson(dim=100, condition_number=1.e3, optimizer='rmsprop', lr=1.e-3, epochs=10000)
        print("Ill conditioned problem - 100d RMSProp + Anderson finished after "+str(len(history))+" iterations "+"\n exact weight sum: 0"+"  - "+" numerical weight sum: "+str(weight_sum))
        self.assertTrue(weight_sum<1e-3)

    def test_100d_ill_conditioned_paraboloid_adam_anderson(self):
        weight_sum, history = test_multiscale_paraboloid_anderson(dim=100, condition_number=1.e3, optimizer='adam', lr=1.e-3, epochs=10000)
        print("Ill conditioned problem - 100d Adam + Anderson finished after "+str(len(history))+" iterations "+"\n exact weight sum: 0"+"  - "+" numerical weight sum: "+str(weight_sum))
        self.assertTrue(weight_sum<1e-3)
        


###############################################################################


if __name__ == "__main__":
    unittest.main()
