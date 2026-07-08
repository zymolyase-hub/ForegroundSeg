import time
import numpy as np


class InferenceTimer:
    """Misst die Laufzeit einer Inferenz."""

    def __init__(self):
        self.start_time = None

    def start(self):
        self.start_time = time.perf_counter()

    def stop(self):
        if self.start_time is None:
            raise RuntimeError("Timer wurde nicht gestartet.")
        return time.perf_counter() - self.start_time


class SegmentationMetrics:
    """Berechnet einfache Kennzahlen für eine Segmentierungsmaske."""

    def __init__(self, mask):

        self.mask = np.asarray(mask)

    @property
    def total_pixels(self):
        return self.mask.size

    @property
    def foreground_pixels(self):
        return np.count_nonzero(self.mask)

    @property
    def background_pixels(self):
        return self.total_pixels - self.foreground_pixels

    @property
    def foreground_ratio(self):
        return self.foreground_pixels / self.total_pixels

    @property
    def background_ratio(self):
        return self.background_pixels / self.total_pixels

    def summary(self):

        return {
            "total_pixels": self.total_pixels,
            "foreground_pixels": self.foreground_pixels,
            "background_pixels": self.background_pixels,
            "foreground_ratio": self.foreground_ratio,
            "background_ratio": self.background_ratio,
        }
