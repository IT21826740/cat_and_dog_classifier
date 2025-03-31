# Cat and Dog Classifier

## Overview
This project is a deep learning-based image classifier that distinguishes between cats and dogs. It uses TensorFlow and Keras to build and train a Convolutional Neural Network (CNN) for classification.

## Dataset
The dataset used for training and testing is available at [Kaggle](https://www.kaggle.com/datasets/dineshpiyasamara/cats-and-dogs-for-classification).

- **Training Data Directory:** `/content/cats-and-dogs-for-classification/cats_dogs/train`
- **Testing Data Directory:** `/content/cats-and-dogs-for-classification/cats_dogs/test`

## Installation
### Requirements
Ensure you have the following dependencies installed:
```sh
pip install tensorflow matplotlib numpy opendatasets
```

### Download Dataset
Use `opendatasets` to download the dataset:
```python
import opendatasets as od
od.download("https://www.kaggle.com/datasets/dineshpiyasamara/cats-and-dogs-for-classification")
```

## Data Preprocessing
The dataset is loaded using `image_dataset_from_directory`, with an 80-10-10 train-validation-test split:
```python
import tensorflow as tf

BATCH_SIZE = 32
IMAGE_SIZE = (128, 128)
SEED = 42

train_data = tf.keras.utils.image_dataset_from_directory(
    train_data_dir, batch_size=BATCH_SIZE, image_size=IMAGE_SIZE,
    subset='training', validation_split=0.1, seed=SEED)

validation_data = tf.keras.utils.image_dataset_from_directory(
    train_data_dir, batch_size=BATCH_SIZE, image_size=IMAGE_SIZE,
    subset='validation', validation_split=0.1, seed=SEED)

test_data = tf.keras.utils.image_dataset_from_directory(
    test_data_dir, batch_size=BATCH_SIZE, image_size=IMAGE_SIZE)
```

### Data Augmentation
To improve generalization, data augmentation is applied:
```python
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal", input_shape=(128,128,3)),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.2),
])
```

## Model Architecture
A CNN model with multiple convolutional and pooling layers:
```python
model = tf.keras.models.Sequential([
    data_augmentation,
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(128, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
```

### Compilation
```python
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
```

## Training the Model
```python
history = model.fit(train_data, epochs=20, validation_data=validation_data)
```

## Performance Analysis
Plot training and validation loss:
```python
import matplotlib.pyplot as plt
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()
```

Plot training and validation accuracy:
```python
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.show()
```

## Model Evaluation
```python
precision = tf.keras.metrics.Precision()
recall = tf.keras.metrics.Recall()
accuracy = tf.keras.metrics.BinaryAccuracy()

for batch in test_data.as_numpy_iterator():
    X, y = batch
    yhat = model.predict(X)
    precision.update_state(y, yhat)
    recall.update_state(y, yhat)
    accuracy.update_state(y, yhat)
```

## Making Predictions
```python
import cv2
img = cv2.imread('/content/cats-and-dogs-for-classification/cats_dogs/test/dogs/dog.4008.jpg')
plt.imshow(img)
plt.show()

yhat = model.predict(img.reshape(1, 128, 128, 3))
print("Dog" if yhat > 0.5 else "Cat")
```

## Repository Structure
```
|-- cats-and-dogs-for-classification
|   |-- cats_dogs
|   |   |-- train
|   |   |   |-- cats
|   |   |   |-- dogs
|   |   |-- test
|   |   |   |-- cats
|   |   |   |-- dogs
|-- notebook.ipynb  # Colab Notebook
|-- README.md       # Project Documentation
```


