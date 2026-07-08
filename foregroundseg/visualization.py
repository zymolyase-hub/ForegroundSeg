from pathlib import Path

import numpy as np
from PIL import Image

import matplotlib.pyplot as plt


class SegmentationVisualizer:
    """
    Utility class for visualizing segmentation results.
    """

    def __init__(self):

        self.foreground_color = np.array([255, 0, 0])
        self.background_color = np.array([0, 0, 255])

    def mask_to_image(self, mask):

        img = np.zeros((mask.shape[0], mask.shape[1], 3),
                       dtype=np.uint8)

        img[mask > 0] = self.foreground_color
        img[mask == 0] = self.background_color

        return Image.fromarray(img)

    def overlay(self, image, mask, alpha=0.35):

        image = np.array(image).astype(np.float32)

        overlay = image.copy()

        overlay[mask > 0] = (
            alpha * self.foreground_color +
            (1 - alpha) * overlay[mask > 0]
        )

        overlay = overlay.astype(np.uint8)

        return Image.fromarray(overlay)

    def foreground(self, image, mask):

        img = np.array(image).copy()

        img[mask == 0] = 0

        return Image.fromarray(img)

    def background(self, image, mask):

        img = np.array(image).copy()

        img[mask > 0] = 0

        return Image.fromarray(img)

    def foreground_png(self, image, mask):

        img = np.array(image)

        rgba = np.zeros((img.shape[0], img.shape[1], 4),
                        dtype=np.uint8)

        rgba[:, :, :3] = img
        rgba[:, :, 3] = np.where(mask > 0, 255, 0)

        return Image.fromarray(rgba)

    def background_png(self, image, mask):

        img = np.array(image)

        rgba = np.zeros((img.shape[0], img.shape[1], 4),
                        dtype=np.uint8)

        rgba[:, :, :3] = img
        rgba[:, :, 3] = np.where(mask == 0, 255, 0)

        return Image.fromarray(rgba)

    def save(self, image, filename):

        Path(filename).parent.mkdir(
            parents=True,
            exist_ok=True
        )

        image.save(filename)

    def comparison_figure(original,
                      overlay,
                      foreground,
                      background):

        fig, ax = plt.subplots(2, 2, figsize=(10, 8))
    
        ax[0,0].imshow(original)
        ax[0,0].set_title("Original")
    
        ax[0,1].imshow(overlay)
        ax[0,1].set_title("Overlay")
    
        ax[1,0].imshow(foreground)
        ax[1,0].set_title("Foreground")
    
        ax[1,1].imshow(background)
        ax[1,1].set_title("Background")
    
        for a in ax.flat:
            a.axis("off")
    
        plt.tight_layout()
    
        return fig
