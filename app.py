import streamlit as st
from deepface import DeepFace
from PIL import Image
import tempfile

st.set_page_config(page_title="Emotion Detector", layout="centered")

st.title("Face Emotion Recognition Demo")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# --- Webcam Input ---
camera_image = st.camera_input("Or take a photo with your webcam")

# Function to analyze image
def analyze_image(img):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        img.save(tmp.name)
        with st.spinner("Analyzing emotion..."):
            try:
                result = DeepFace.analyze(img_path=tmp.name, actions=["emotion"])
                emotion = result[0]['dominant_emotion']
                scores = result[0]['emotion']

                # Map to emojis for fun
                emoji_map = {
                    "happy": "😄",
                    "sad": "😢",
                    "angry": "😠",
                    "surprise": "😲",
                    "fear": "😨",
                    "neutral": "😐"
                }

                st.success(f"{emoji_map.get(emotion, '')} Detected Emotion: {emotion}")
                st.subheader("Emotion Scores")
                st.json(scores)
            except Exception as e:
                st.error("No face detected or error occurred")
                st.text(str(e))

# Use uploaded image if available, else webcam
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width="stretch")
    analyze_image(image)
elif camera_image:
    image = Image.open(camera_image)
    st.image(image, caption="Webcam Image", width="stretch)
    analyze_image(image)
else:
    st.info("Upload an image or use your webcam to detect emotion")


# if uploaded_file:
#     image = Image.open(uploaded_file)
#     st.image(image, caption="Uploaded Image", width="stretch")

#     with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
#         image.save(tmp.name)

#         with st.spinner("Analyzing emotion..."):
#             result = DeepFace.analyze(img_path=tmp.name, actions=["emotion"])

#         emotion = result[0]['dominant_emotion']
#         scores = result[0]['emotion']

#         st.success(f"Detected Emotion: {emotion}")

#         st.subheader("Emotion Scores")
#         st.json(scores)