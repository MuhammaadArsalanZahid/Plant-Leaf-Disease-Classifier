import os
import json

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
# Force PyTorch to download/cache pretrained weights on the D drive so the
# system drive (C:) is not filled up during model initialization.
os.environ["TORCH_HOME"] = "D:/pytorch_cache"

import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

from src.model import get_transfer_model


# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Plant Leaf Disease Classifier",
    page_icon="🌿",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.title("🌿 Plant Leaf Disease Classifier")
st.markdown(
    "Upload a tomato leaf photo to detect its health condition. "
    "Powered by a MobileNetV2 transfer-learning model."
)
st.markdown("---")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CLASSES = ["Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___healthy"]
MODEL_PATH = "models/best_model.pth"
TIPS_PATH = "assets/disease_tips.json"
CONFIDENCE_THRESHOLD = 0.65

# Ensure the model directory exists (used when saving trained checkpoints).
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)


# ---------------------------------------------------------------------------
# Model & tips loading
# ---------------------------------------------------------------------------
@st.cache_resource
def load_classifier():
    model = get_transfer_model(num_classes=len(CLASSES), freeze_base=False)
    if os.path.exists(MODEL_PATH):
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
        except Exception as exc:  # noqa: BLE001
            st.error(
                f"Unable to load trained weights ({exc}). "
                "Running in initialization mode with random weights."
            )
    model.eval()
    return model


@st.cache_data
def load_tips():
    if os.path.exists(TIPS_PATH):
        with open(TIPS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


model = load_classifier()
tips_dict = load_tips()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.header("📁 Navigation & Options")
st.sidebar.caption("Plant Leaf Disease Classifier · Tomato")
uploaded_file = st.sidebar.file_uploader("Upload Leaf Photo", type=["jpg", "jpeg", "png"])
st.sidebar.markdown("---")
st.sidebar.markdown("**Confidence threshold:** 65%")
st.sidebar.markdown(
    "Predictions below this threshold are flagged for review. "
    "Upload a clear, well-lit close-up of the leaf for best results."
)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def predict_image(image):
    preprocess = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    tensor_img = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        output = model(tensor_img)
        probabilities = F.softmax(output, dim=1)
    top_prob, top_idx = torch.topk(probabilities, min(3, len(CLASSES)))
    return [(CLASSES[idx], prob.item()) for prob, idx in zip(top_prob[0], top_idx[0])]


# ---------------------------------------------------------------------------
# Main flow
# ---------------------------------------------------------------------------
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    col_img, col_meta = st.columns([2, 1])
    with col_img:
        st.image(image, caption="Uploaded Leaf Specimen", use_container_width=True)
    with col_meta:
        st.success("File uploaded successfully. Analyzing...")

    with st.spinner("Analyzing leaf patterns..."):
        predictions = predict_image(image)
        top_class, top_conf = predictions[0]

    st.markdown("## 🩺 Diagnosis Result")

    if top_conf < CONFIDENCE_THRESHOLD:
        st.warning(
            "⚠️ **Confidence low.** The model is not entirely confident "
            "in identifying this condition. Please provide a clearer "
            "close-up photograph of the affected leaf under optimal lighting."
        )
    else:
        st.success(f"**Primary Diagnosis:** {top_class.replace('___', ' - ')}")

    st.markdown("### 📊 Top Predictions")
    for cls, score in predictions:
        label = cls.replace("___", " - ")
        st.progress(score, text=f"{label}: {score * 100:.1f}%")

    st.markdown("### 💡 Expert Agricultural Recommendation")
    tip = tips_dict.get(
        top_class,
        "No specific advice found for this category. Monitor growth metrics closely.",
    )
    st.info(f"**Treatment Strategy:** {tip}")
else:
    st.info(
        "💡 **Usage tip:** Upload a clean plant leaf image from the sidebar "
        "panel to check its health."
    )

st.markdown("---")
st.caption(
    "Demonstration model. For serious agricultural decisions, please consult "
    "a qualified plant pathologist or agronomist."
)
