"""Streamlit application for the FashionMNIST image classifier."""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms


class SimpleCNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, 3, 1)
        self.conv2 = torch.nn.Conv2d(32, 64, 3, 1)
        self.fc1 = torch.nn.Linear(9216, 128)
        self.fc2 = torch.nn.Linear(128, 10)
        self.activations = None

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        self.activations = x
        if x.requires_grad:
            x.retain_grad()
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


CLASSES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
           "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
MODEL_PATH = Path(__file__).resolve().parent / "fashion_mnist_cnn.pth"


@st.cache_resource
def load_model():
    model = SimpleCNN()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model


TRANSFORM = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])


def generate_gradcam(input_tensor, model, class_idx):
    """Generate Grad-CAM using gradients of the final convolutional features."""
    model.zero_grad(set_to_none=True)
    output = model(input_tensor)
    output[0, class_idx].backward()

    activations = model.activations.detach()[0]
    gradients = model.activations.grad.detach()[0]
    weights = gradients.mean(dim=(1, 2), keepdim=True)
    cam = torch.relu((weights * activations).sum(dim=0))

    max_value = cam.max()
    if max_value.item() > 0:
        cam = cam / max_value
    return cam.cpu().numpy()


st.set_page_config(page_title="Fashion Image Classifier", layout="centered")
st.title("🧠 Fashion Image Classifier with Explainable AI")
st.markdown("Upload a clothing image to classify it and view an explanation heatmap.")
st.info("FashionMNIST uses small 28×28 grayscale catalogue images. Real-world photos may be outside the model's training distribution.")

model = load_model()
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image", use_column_width=True)
    input_tensor = TRANSFORM(image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        pred_idx = output.argmax(dim=1).item()
        confidence = torch.softmax(output, dim=1)[0, pred_idx].item()

    st.markdown(f"**Predicted class:** {CLASSES[pred_idx]}")
    st.markdown(f"**Confidence:** {confidence:.1%}")

    cam = generate_gradcam(input_tensor, model, pred_idx)
    cam = cv2.resize(cam, (28, 28))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    original = np.array(image.resize((28, 28)).convert("L"))
    original = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
    overlay = cv2.addWeighted(original, 0.5, heatmap, 0.5, 0)

    st.markdown("### Grad-CAM heatmap")
    st.image(overlay, channels="BGR", caption="Class-specific activation overlay", use_column_width=True)
    st.caption("CNN trained on FashionMNIST | PyTorch + Streamlit + Grad-CAM")
