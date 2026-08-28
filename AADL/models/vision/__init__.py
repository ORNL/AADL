"""Vision architectures and an explicit model factory."""

from .densenet import DenseNet121, DenseNet161, DenseNet169, DenseNet201
from .dla import DLA
from .dla_simple import SimpleDLA
from .dpn import DPN26, DPN92
from .efficientnet import EfficientNetB0
from .googlenet import GoogLeNet
from .lenet import LeNet
from .mobilenet import MobileNet
from .mobilenetv2 import MobileNetV2
from .pnasnet import PNASNetA, PNASNetB
from .preact_resnet import (
    PreActResNet18,
    PreActResNet34,
    PreActResNet50,
    PreActResNet101,
    PreActResNet152,
)
from .regnet import RegNetX_200MF, RegNetX_400MF, RegNetY_400MF
from .resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from .resnext import (
    ResNeXt29_2x64d,
    ResNeXt29_4x64d,
    ResNeXt29_8x64d,
    ResNeXt29_32x4d,
)
from .senet import SENet18
from .shufflenet import ShuffleNetG2, ShuffleNetG3
from .shufflenetv2 import ShuffleNetV2
from .vgg import VGG


MODEL_REGISTRY = {
    "densenet121": DenseNet121,
    "densenet161": DenseNet161,
    "densenet169": DenseNet169,
    "densenet201": DenseNet201,
    "dpn26": DPN26,
    "dpn92": DPN92,
    "efficientnet_b0": EfficientNetB0,
    "googlenet": GoogLeNet,
    "lenet": LeNet,
    "mobilenet": MobileNet,
    "mobilenet_v2": MobileNetV2,
    "pnasnet_a": PNASNetA,
    "pnasnet_b": PNASNetB,
    "preact_resnet18": PreActResNet18,
    "preact_resnet34": PreActResNet34,
    "preact_resnet50": PreActResNet50,
    "preact_resnet101": PreActResNet101,
    "preact_resnet152": PreActResNet152,
    "regnet_x_200mf": RegNetX_200MF,
    "regnet_x_400mf": RegNetX_400MF,
    "regnet_y_400mf": RegNetY_400MF,
    "resnet18": ResNet18,
    "resnet34": ResNet34,
    "resnet50": ResNet50,
    "resnet101": ResNet101,
    "resnet152": ResNet152,
    "resnext29_2x64d": ResNeXt29_2x64d,
    "resnext29_4x64d": ResNeXt29_4x64d,
    "resnext29_8x64d": ResNeXt29_8x64d,
    "resnext29_32x4d": ResNeXt29_32x4d,
    "senet18": SENet18,
    "shufflenet_g2": ShuffleNetG2,
    "shufflenet_g3": ShuffleNetG3,
    "simple_dla": SimpleDLA,
}


def create_model(name, **kwargs):
    """Construct a registered vision model by its case-insensitive name."""
    key = name.lower()
    try:
        constructor = MODEL_REGISTRY[key]
    except KeyError as exc:
        available = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"unknown model {name!r}; available models: {available}") from exc
    return constructor(**kwargs)


def list_models():
    """Return the registered model names in deterministic order."""
    return tuple(sorted(MODEL_REGISTRY))


__all__ = [
    "MODEL_REGISTRY",
    "create_model",
    "list_models",
    *[constructor.__name__ for constructor in MODEL_REGISTRY.values()],
]
