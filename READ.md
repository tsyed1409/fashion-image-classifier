# Fashion Image Classifier
# 🧠 Fashion Item Classifier with Grad-CAM

This AI-powered Streamlit app classifies fashion item images using a trained convolutional neural network (CNN) on the FashionMNIST dataset. It also uses Grad-CAM to visualize which parts of the image the model focused on.

🔗 [Live Demo](https://tsyed1409-fashion-image-classifier.streamlit.app)

---

## 💡 Features

- Upload an image of a fashion item (e.g., shirt, shoe, coat)
- Get a prediction with label (e.g., Sneaker, T-shirt, etc.)
- View Grad-CAM heatmap explaining what the model looked at

---

## 🧠 Model Details

- Dataset: FashionMNIST (60,000 training images)
- Architecture: Simple CNN (Conv2D + ReLU + MaxPool + FC layers)
- Explainability: Grad-CAM over final convolutional layer

---

## 🗂 Files

- `app.py` – Streamlit frontend and model inference code
- `fashion_mnist_cnn.pth` – Pretrained model weights
- `requirements.txt` – Python dependencies

---

## 🧪 Try It Yourself

```bash
# Clone this repo
git clone https://github.com/tsyed1409/fashion-image-classifier.git
cd fashion-image-classifier

# Install dependencies
pip install -r requirements.txt

# Run locally
streamlit run app.py
