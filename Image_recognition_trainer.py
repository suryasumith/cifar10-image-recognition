# IMAGE RECOGNITION USING CIFAR-10
# FILE: Image_Reconition_Trainer.py
# DESCRIPTION: Trains a CNN model on CIFAR-10 and saves it
# Author: U. VAMSHI | Roll No: 1009-22-861-022
# Nizam College (Autonomous), Dept. of Informatics
# ============================================================

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.utils import to_categorical
import h5py

# ── Step 1: Load CIFAR-10 Dataset ──────────────────────────
print("Loading CIFAR-10 dataset...")
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# ── Step 2: Normalize pixel values (0-255 → 0.0-1.0) ──────
new_X_train = X_train.astype('float32')
new_X_test  = X_test.astype('float32')
new_X_train /= 255
new_X_test  /= 255

# ── Step 3: One-hot encode labels ──────────────────────────
new_Y_train = to_categorical(y_train)
new_Y_test  = to_categorical(y_test)

# ── Step 4: Build the CNN Model ────────────────────────────
print("Building CNN model...")
model = Sequential()

# Conv Layer 1 — 32 filters, 3x3 kernel, ReLU activation
model.add(Conv2D(32, (3, 3),
                 input_shape=(32, 32, 3),
                 activation='relu',
                 padding='same',
                kernel_constraint=max_norm(3)))

# Max Pooling Layer 1
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten the feature maps
model.add(Flatten())
# Fully Connected Layer — 512 neurons
model.add(Dense(512, activation='relu', kernel_constraint=max_norm(3)))

# Dropout for regularization (prevents overfitting)
model.add(Dropout(0.5))

# Output Layer — 10 neurons (one per CIFAR-10 class)
model.add(Dense(10, activation='softmax'))

# ── Step 5: Compile the Model ──────────────────────────────
model.compile(
    loss='categorical_crossentropy',
    optimizer=SGD(learning_rate=0.01),
    metrics=['accuracy']
)

model.summary()

# ── Step 6: Train the Model ────────────────────────────────
print("\nTraining the model... (This may take a few minutes)")
model.fit(
    new_X_train, new_Y_train,
    epochs=10,
    batch_size=32,
    validation_data=(new_X_test, new_Y_test),
    verbose=1
)

# ── Step 7: Evaluate Accuracy ──────────────────────────────
scores = model.evaluate(new_X_test, new_Y_test, verbose=0)
print(f"\nTest Accuracy: {scores[1] * 100:.2f}%")

# ── Step 8: Save the Trained Model ────────────────────────
model.save('trained_model.h5')
print("\nModel saved as 'trained_model.h5'")
print("Now run 'Image_Recognition_Tester.py' to classify your own images!")
