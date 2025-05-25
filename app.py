# app.py – Streamlit UI for Image Classifier with Grad-CAM

import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load model class again
class SimpleCNN(torch.nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, 3, 1)
        self.conv2 = torch.nn.Conv2d(32, 64, 3, 1)
        self.fc1 = torch.nn.Linear(9216, 128)
        self.fc2 = torch.nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        self.activations = x  # Grad-CAM hook
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Class labels
classes = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat",
           "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot"]

# Load trained model
model = SimpleCNN()
model.load_state_dict(torch.load("fashion_mnist_cnn.pth", map_location=torch.device('cpu')))
model.eval()

# Image transforms
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Grad-CAM function
def generate_gradcam(input_tensor, model, class_idx):
    model.zero_grad()
    output = model(input_tensor)
    score = output[0, class_idx]
    score.backward()
    gradients = model.conv2.weight.grad
    activations = model.activations.detach().squeeze()
    weights = gradients.mean(dim=[2, 3])[0]

    cam = torch.zeros(activations.shape[1:])
    for i, w in enumerate(weights):
        cam += w * activations[i]
    cam = cam.numpy()
    cam = np.maximum(cam, 0)
    cam = cam / cam.max()
    return cam

# Streamlit UI
st.set_page_config(layout="centered")
st.title("🧠 Fashion Item Classifier with Grad-CAM")
st.markdown("Upload a clothing image. The AI model will predict its class and explain what it focused on.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png"])

if uploaded_file:
    # Load image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess for model
    input_tensor = transform(image).unsqueeze(0)
    output = model(input_tensor)
    pred_idx = output.argmax(dim=1).item()
    confidence = torch.softmax(output, dim=1)[0][pred_idx].item()
    st.markdown(f"**Predicted Class:** {classes[pred_idx]}  ")
    st.markdown(f"**Confidence:** {confidence:.2f}")

    # Grad-CAM
    cam = generate_gradcam(input_tensor, model, pred_idx)
    cam = cv2.resize(cam, (28, 28))
    cam = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    orig_img = np.array(image.resize((28, 28)).convert("L"))
    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_GRAY2BGR)
    overlay = cv2.addWeighted(orig_img, 0.5, cam, 0.5, 0)

    st.markdown("### Grad-CAM Heatmap")
    st.image(overlay, channels="BGR", caption="Grad-CAM Overlay", use_column_width=True)

    st.caption("Model: CNN trained on FashionMNIST | Built with PyTorch + Streamlit")
