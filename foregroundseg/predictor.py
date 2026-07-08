from torchvision.models.segmentation import deeplabv3_resnet101

import torch

from .preprocessing import ImagePreprocessor


class ForegroundPredictor:

    def __init__(self):

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model = deeplabv3_resnet101(
            weights="DEFAULT"
        )

        self.model.to(self.device)

        self.model.eval()

        self.preprocessor = ImagePreprocessor()

    def predict(self, image):

        tensor = self.preprocessor.preprocess(image)

        tensor = tensor.to(self.device)

        with torch.no_grad():

            prediction = self.model(tensor)["out"]

        return prediction.argmax(1).squeeze().cpu().numpy()
