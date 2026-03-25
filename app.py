import streamlit as st
from deepface import DeepFace
from PIL import Image
import tempfile

st.set_page_config(page_title="Emotion Detector", layout="centered")

st.title("😊 Face Emotion Recognition Demo")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        image.save(tmp.name)

        with st.spinner("Analyzing emotion..."):
            result = DeepFace.analyze(img_path=tmp.name, actions=["emotion"])

        emotion = result[0]['dominant_emotion']
        scores = result[0]['emotion']

        st.success(f"Detected Emotion: {emotion}")

        st.subheader("Emotion Scores")
        st.json(scores)