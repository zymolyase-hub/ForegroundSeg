from dataclasses import dataclass

import torch

from torchvision.models.segmentation import (
    deeplabv3_resnet101,
    deeplabv3_resnet50,
    fcn_resnet101,
    lraspp_mobilenet_v3_large,
)


@dataclass
class LoadedModel:
    """
    Container für ein geladenes Modell.
    """

    model: torch.nn.Module
    device: torch.device
    name: str


class ModelLoader:
    """
    Lädt Segmentierungsmodelle aus TorchVision.
    """

    SUPPORTED_MODELS = {
        "deeplabv3_resnet101": deeplabv3_resnet101,
        "deeplabv3_resnet50": deeplabv3_resnet50,
        "fcn_resnet101": fcn_resnet101,
        "lraspp_mobilenet_v3_large": lraspp_mobilenet_v3_large,
    }

    def __init__(self, device=None):

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)

    def load(self, model_name="deeplabv3_resnet101"):

        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Unbekanntes Modell: {model_name}"
            )

        model_fn = self.SUPPORTED_MODELS[model_name]

        model = model_fn(weights="DEFAULT")

        model.to(self.device)

        model.eval()

        return LoadedModel(
            model=model,
            device=self.device,
            name=model_name,
        )
