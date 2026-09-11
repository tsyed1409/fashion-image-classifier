# Fashion Image Classifier with Explainable AI

An end-to-end computer vision project demonstrating supervised model training, inference and model explainability. A convolutional neural network (CNN) is trained on FashionMNIST using PyTorch and deployed through an interactive Streamlit application.

**Live demo:** https://tsyed1409-fashion-image-classifier.streamlit.app

## What it does

The application accepts a fashion image, converts it to the 28 x 28 grayscale format used by FashionMNIST, runs inference through a trained CNN, displays the predicted class and confidence score, and generates an explanation heatmap.

The ten supported FashionMNIST classes are T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag and Ankle boot.

## Why I built it

This project demonstrates the lifecycle of a small machine-learning solution rather than simply consuming a pre-trained AI API: labelled training data, neural-network architecture, model training and evaluation, persisted model weights, application inference and an explainability layer.

## Technology

Python, PyTorch, Torchvision, FashionMNIST, Streamlit, OpenCV and NumPy.

## Architecture

```text
FashionMNIST -> train.py -> PyTorch CNN -> saved model weights
                                             |
                                             v
                                         app.py
                                      /          \
                               prediction      heatmap
```

## Model

`SimpleCNN` uses two convolutional layers with ReLU activations, max pooling and two fully connected layers. The final layer produces logits for the ten FashionMNIST classes.

The included `train.py` provides a reproducible training pipeline. It downloads FashionMNIST, trains a compatible CNN for five epochs using Adam and cross-entropy loss, reports test accuracy after each epoch and saves the resulting state dictionary.

> The committed `.pth` file contains previously trained weights. The training script provides a reproducible compatible pipeline; exact retraining results can vary by environment and hardware.

## Run locally

```bash
git clone https://github.com/tsyed1409/fashion-image-classifier.git
cd fashion-image-classifier
python -m venv .venv
pip install -r requirements.txt
streamlit run app.py
```

## Retrain the model

```bash
python train.py
```

FashionMNIST will be downloaded into `data/` and the trained weights written to `fashion_mnist_cnn.pth`.

## Repository structure

```text
README.md               Project documentation
app.py                  Streamlit inference and explainability UI
train.py                Reproducible model-training pipeline
fashion_mnist_cnn.pth   Trained model weights
requirements.txt        Python dependencies
.gitignore              Local and generated files excluded from Git
```

## Explainability

The application includes a visual activation heatmap intended to demonstrate explainable-AI principles by highlighting image regions associated with a prediction. Explainability visualisations are aids to interpretation and should not be treated as complete causal explanations of model behaviour.

## Limitations

FashionMNIST contains small 28 x 28 grayscale catalogue-style images. Arbitrary real-world photographs can differ substantially from the training distribution, so predictions and confidence scores may be unreliable for such images. This is a demonstration project, not a production fashion-recognition system.

## Future improvements

Potential next steps include a validation split, confusion matrix and per-class metrics, automated tests and CI, model/version metadata, data augmentation, and comparison with transfer-learning architectures.

## Author

**Tariq Syed** — AI, product and digital transformation practitioner.
