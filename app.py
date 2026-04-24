import streamlit as st
from PIL import Image
from main import MultimodalFakeNewsDetector



st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://www.image2url.com/r2/default/images/1776875440858-b189c852-7482-4ab1-8e63-bbfa2ad9e0fa.jpg");
        background-size:100%;  
        background-position: center upper;
        background-attachment: fixed;
        margin-top:-20px;
    }

    /* Make text readable */
    .stTextInput, .stTextArea, .stFileUploader {
        background-color: rgba(255, 255, 255, 0.85);
        border-radius: 10px;
        padding: 10px;
    }

    /* Button styling */
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 10px;
        height: 50px;
        width: 200px;
        font-size: 18px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load model
detector = MultimodalFakeNewsDetector()
detector.train("train.csv","images/")

st.markdown("<h1 style='text-align:center; font-size:48px; color:#FFFF00; font-family:bebas neue, sans-serif; text-shadow:2px 2px 10px red;'>🌍FAKE NEWS DETECTION USING MULTI MODAL AI </h1>", unsafe_allow_html=True)

# Input text
text = st.text_area("Enter News Text")

# Upload image
image_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
if st.button("Predict"):
    result = detector.predict(text, image_file)

    result_str = str(result).lower()

    if "real" in result_str or result == 1:
        st.markdown("<h2 style='color:#FFFF00;'>✅ Real News</h2>", unsafe_allow_html=True)
    else:
        st.markdown("<h2 style='color:#ff4b4b;'>❌ Fake News</h2>", unsafe_allow_html=True)

if st.button("Check News"):

    if text and image_file:
        image = Image.open(image_file)

        # Save temp image
        image_path = "temp.jpg"
        image.save(image_path)

        text_lower = text.lower()

        # Same logic as backend
        if any(word in text_lower for word in ["alien", "ufo", "ghost", "miracle", "hoax"]):
            st.error("🔴 FAKE NEWS")
        elif any(word in text_lower for word in ["government", "official", "report", "policy", "scientists"]):
            st.success("🟢 REAL NEWS")
        else:
            result = detector.predict(text, image_path)
            st.info("Prediction completed")

    else:
        st.warning("Please enter text and upload image")