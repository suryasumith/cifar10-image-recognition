
# CIFAR-10 Image Recognition System

An end-to-end deep learning project designed to classify images from the CIFAR-10 dataset into 10 distinct classes using a Convolutional Neural Network (CNN).

---

## 📁 Repository Files

* `image_recognition_trainer.py`: Loads the CIFAR-10 dataset, performs image normalization, trains the CNN model, and exports the trained weights.
* `image_recognition_tester.py`: Imports the trained model weights and executes inference on test images to evaluate classification accuracy.

---

## ⚙️ Environment Setup & Installation

### 1. Clone or Download Repository
```bash
git clone [https://github.com/suryasumith/cifar10-image-recognition.git](https://github.com/suryasumith/cifar10-image-recognition.git)
cd cifar10-image-recognition

steps to process and run the programme
Install Dependencies - pip install tensorflow numpy matplotlib

Step-by-Step Execution Guide

Step 1: Model Training
python image_recognition_trainer.py

Step 2: Model Testing & Verification
python image_recognition_tester.py
