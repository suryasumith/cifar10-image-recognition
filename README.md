# CIFAR-10 Image Recognition System

An end-to-end deep learning project that trains a Convolutional Neural Network (CNN) on the CIFAR-10 dataset and provides interactive image classification inference.

---

## 📁 Repository Overview

* `Image_recognition_trainer.py`: Loads the CIFAR-10 dataset, performs pixel normalization, constructs the CNN model, trains for 10 epochs, and exports `trained_model.h5`.
* `image_recognition_tester.py`: Loads `trained_model.h5`, takes an external image file path, resizes it to 32x32 RGB, and predicts the class label along with percentage confidence scores.

---

## ⚙️ Setup & Dependencies

Make sure Python (3.8+) is installed. Install the necessary packages via terminal/command prompt:

```bash
pip install tensorflow numpy pillow h5py

-----Execution Guide-------

Step 1: Train & Save the Model
python Image_recognition_trainer.py

Step 2: Test Single Images (Interactive Classification)
python image_recognition_tester.py

When prompted in the terminal:
Enter image file pathname: path/to/your/image.jpg
