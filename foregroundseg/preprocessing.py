from PIL import Image

import torchvision.transforms as transforms


class ImagePreprocessor:
    """
    Converts PIL images into tensors that can be used
    by the segmentation network.
    """

    def __init__(self, image_size=None):

        transform_list = []

        if image_size is not None:
            transform_list.append(
                transforms.Resize(image_size)
            )

        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        self.transform = transforms.Compose(transform_list)

    def preprocess(self, image: Image.Image):

        if image.mode != "RGB":
            image = image.convert("RGB")

        tensor = self.transform(image)

        return tensor.unsqueeze(0)
