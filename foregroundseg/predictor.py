from .models.model_loader import ModelLoader
from .preprocessing import ImagePreprocessor


class ForegroundPredictor:

    def __init__(self,
                 model_name="deeplabv3_resnet101"):

        loader = ModelLoader()

        loaded = loader.load(model_name)

        self.model = loaded.model
        self.device = loaded.device
        self.model_name = loaded.name

        self.preprocessor = ImagePreprocessor()

    def predict(self, image):

        tensor = self.preprocessor.preprocess(image)

        tensor = tensor.to(self.device)

        with torch.no_grad():

            output = self.model(tensor)["out"]

        return output.argmax(1).squeeze().cpu().numpy()
