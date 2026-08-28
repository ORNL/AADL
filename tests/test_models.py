import unittest

import torch

from AADL.models import LinearRegression, MLP
from AADL.models.vision import create_model, list_models
from tests.fixtures.models import Paraboloid, Rosenbrock


class ModelPackageTests(unittest.TestCase):
    def test_basic_models_are_package_importable(self):
        regression = LinearRegression(1, 1)
        mlp = MLP(1, 1, [2], True, "relu")

        self.assertEqual(regression(torch.ones(1, 1)).shape, (1, 1))
        self.assertEqual(mlp(torch.ones(1, 1)).shape, (1,))

    def test_test_models_live_in_fixtures(self):
        self.assertEqual(Rosenbrock(2, initial_guess=[1, 1])(None).item(), 0.0)
        self.assertEqual(Paraboloid(2, condition_number=1).get_weight().shape, (2,))

    def test_vision_registry_constructs_model(self):
        self.assertIn("resnet18", list_models())
        model = create_model("resnet18", num_classes=7)
        self.assertEqual(model(torch.randn(1, 3, 32, 32)).shape, (1, 7))

    def test_vision_registry_rejects_unknown_name(self):
        with self.assertRaisesRegex(ValueError, "unknown model"):
            create_model("not-a-model")


if __name__ == "__main__":
    unittest.main()
