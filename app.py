import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
from model import Model

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Cancer Classification CNN",
    page_icon="🩺",
    layout="centered"
)

# -----------------------------
# Title
# -----------------------------
st.title("🩺 Lung & Colon Cancer Classification")
st.markdown(
    """
    Deep Learning-based histopathological image classification  
    using a custom Convolutional Neural Network (CNN).
    """
)

st.divider()

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("📌 About")
st.sidebar.write(
    """
    This application classifies histopathological tissue images into 5 categories:

    - Colon Adenocarcinoma
    - Colon Benign
    - Lung Adenocarcinoma
    - Lung Benign
    - Lung Squamous Cell Carcinoma
    """
)

st.sidebar.header("⚙️ Model")
st.sidebar.write(
    """
    - Custom CNN (PyTorch)
    - Test Accuracy: 96.16%
    - Input Size: 128×128 RGB
    """
)

# -----------------------------
# Class Names
# -----------------------------
classes = [
    "Colon Adenocarcinoma",
    "Colon Benign",
    "Lung Adenocarcinoma",
    "Lung Benign",
    "Lung Squamous Cell Carcinoma"
]

# -----------------------------
# Device
# -----------------------------
device = torch.device("cpu")

# -----------------------------
# Load Model
# -----------------------------
model = Model()
model.load_state_dict(torch.load("best.pth", map_location=device))
model.eval()

# -----------------------------
# Transform
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5,0.5,0.5],
        std=[0.5,0.5,0.5]
    )
])

# -----------------------------
# Upload Section
# -----------------------------
uploaded_file = st.file_uploader(
    "📤 Upload Histopathology Image",
    type=["jpg","jpeg","png"]
)

# -----------------------------
# Prediction
# -----------------------------
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    img = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img)
        probabilities = torch.softmax(outputs, dim=1)

    confidence, predicted = torch.max(probabilities,1)

    st.divider()

    # Prediction Box
    st.success(
        f"### Prediction: {classes[predicted.item()]}"
    )

    st.info(
        f"Confidence: {confidence.item()*100:.2f}%"
    )

    # Probability Scores
    st.subheader("📊 Class Probabilities")

    probs = probabilities.squeeze().tolist()

    for i, cls in enumerate(classes):
        st.write(f"{cls}")
        st.progress(float(probs[i]))

st.divider()

# Footer
st.caption(
    "Developed by Abhinav Singh | IIT Dharwad"
)