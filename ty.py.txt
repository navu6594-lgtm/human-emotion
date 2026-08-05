import streamlit as st
from transformers import pipeline
from PIL import Image
import io
import numpy as np

st.title("📷 Emotion Detector + Comfort Flow (Double Snapshot)")

# Load HuggingFace models
@st.cache_resource
def load_image_model():
    return pipeline("image-classification", model="trpakov/vit-face-expression")

@st.cache_resource
def load_text_model():
    return pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")

image_model = load_image_model()
text_model = load_text_model()

# Step 1: First camera snapshot
img_file1 = st.camera_input("Take your first picture")

if img_file1 is not None:
    # Convert UploadedFile to PIL image safely
    img_bytes1 = img_file1.getvalue()
    img1 = Image.open(io.BytesIO(img_bytes1)).convert("RGB")

    # Show image as NumPy array
    st.image(np.array(img1), caption="First snapshot", use_column_width=True)

    # Detect emotion from first snapshot
    results1 = image_model(img1)
    emotion1 = results1[0]['label']
    st.subheader("First Detected Emotion:")
    st.write(f"**{emotion1}** (confidence {results1[0]['score']:.2f})")

    # Step 2: Ask why
    reason = st.text_input(f"Why are you feeling {emotion1.lower()}?")

    if reason:
        # Step 3: Second camera snapshot
        img_file2 = st.camera_input("Take another picture after replying")

        if img_file2 is not None:
            img_bytes2 = img_file2.getvalue()
            img2 = Image.open(io.BytesIO(img_bytes2)).convert("RGB")

            st.image(np.array(img2), caption="Second snapshot", use_column_width=True)

            # Detect emotion from second snapshot
            results2 = image_model(img2)
            emotion2 = results2[0]['label']

            # Detect emotion in reply text
            reply_results = text_model(reason)
            reply_emotion = reply_results[0]['label']

            # Step 4: Comfort message
            st.success(
                f"Thanks for sharing. I noticed your face shows {emotion2.lower()} "
                f"and your words express {reply_emotion.lower()}. "
                f"Remember, it's okay to feel {emotion1.lower()} — you're not alone 💙"
            )
