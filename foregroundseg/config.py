from pathlib import Path

PACKAGE_ROOT = Path(__file__).parent
PROJECT_ROOT = PACKAGE_ROOT.parent

MODEL_NAME = "DeepLabV3"

DEFAULT_DEVICE = "cuda"

EXAMPLES_DIR = PROJECT_ROOT / "examples"

OUTPUT_DIR = PROJECT_ROOT / "output"
