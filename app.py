import time
import streamlit as st
from PIL import Image

from foregroundseg.predictor import ForegroundPredictor
from foregroundseg.image_utils import (
    create_foreground_mask,
    extract_foreground,
    extract_background,
    overlay,
)

st.set_page_config(
    page_title="ForegroundSeg",
    layout="wide"
)

st.title("ForegroundSeg")
st.subheader("AI Foreground / Background Segmentation with PyTorch")

uploaded = st.file_uploader(
    "Upload an image",
    type=["png", "jpg", "jpeg"]
)

if uploaded:

    image = Image.open(uploaded).convert("RGB")

    predictor = ForegroundPredictor()

    start = time.time()

    segmentation = predictor.predict(image)

    elapsed = time.time() - start

    mask = create_foreground_mask(segmentation)

    foreground = extract_foreground(image, mask)
    background = extract_background(image, mask)
    overlay_img = overlay(image, mask)

    c1, c2 = st.columns(2)

    with c1:
        st.image(image, caption="Original")
        st.image(overlay_img, caption="Overlay")

    with c2:
        st.image(foreground, caption="Foreground")
        st.image(background, caption="Background")

    st.metric(
        "Inference Time",
        f"{elapsed:.2f} s"
    )

    foreground_pixels = (mask > 0).sum()
    total_pixels = mask.size

    st.metric(
        "Foreground",
        f"{100*foreground_pixels/total_pixels:.1f}%"
    )

    st.metric(
        "Background",
        f"{100*(1-foreground_pixels/total_pixels):.1f}%"
    )
