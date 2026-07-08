from predictor import ForegroundPredictor

def test_model_loads():
    model = ForegroundPredictor()
    assert model is not None
