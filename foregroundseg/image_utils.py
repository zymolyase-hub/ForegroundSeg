import numpy as np
from PIL import Image

# Klassen >0 gelten hier vereinfachend als Vordergrund.
# TODO: Für eine produktive Anwendung gezielt die gewünschten Klassen
# (z. B. Person, Hund, Katze, Auto ...) auswählen können.

def create_foreground_mask(segmentation):

    mask = np.where(segmentation > 0, 255, 0).astype(np.uint8)

    return mask


def extract_foreground(image, mask):

    img = np.array(image)

    foreground = img.copy()
    foreground[mask == 0] = 0

    return Image.fromarray(foreground)


def extract_background(image, mask):

    img = np.array(image)

    background = img.copy()
    background[mask == 255] = 0

    return Image.fromarray(background)


def overlay(image, mask):

    img = np.array(image).copy()

    red = np.zeros_like(img)
    red[:, :, 0] = 255

    alpha = 0.35

    img[mask == 255] = (
        alpha * red[mask == 255]
        + (1 - alpha) * img[mask == 255]
    )

    return Image.fromarray(img.astype(np.uint8))
