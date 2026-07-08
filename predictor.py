import torch
import torchvision.transforms as transforms
from torchvision.models.segmentation import deeplabv3_resnet101
from PIL import Image
import numpy as np

class ForegroundPredictor:

    def __init__(self):

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model = deeplabv3_resnet101(weights="DEFAULT")
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485,0.456,0.406],
                std=[0.229,0.224,0.225]
            )
        ])

    def predict(self, image: Image.Image):

        tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(tensor)["out"]

        prediction = output.argmax(1).squeeze().cpu().numpy()

        return prediction
