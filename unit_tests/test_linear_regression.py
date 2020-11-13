from modules.NN_models import MLP
from modules.optimizers import FixedPointIteration

def test_sum():
    input_dim, output_dim, dataset = ...
    assert sum([1, 2, 3]) == 6, "Should be 6"

if __name__ == "__main__":
    test_sum()
    print("Everything passed")
