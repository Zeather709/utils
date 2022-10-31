# Convolutional Neural Network

# Import libraries

import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

tf.__version__

# Data Preprocessing

# Preprocessing training set
# Apply transformation to training data set to avoid overfitting
# Called image augmentation

train_datagen = ImageDataGenerator(
    rescale=1. / 255,  # Normalization - get value of all pixels between 0 and 1 (feature scaling)
    zoom_range=0.2,
    horizontal_flip=True)

train = train_datagen.flow_from_directory(
    '/home/zeather/data/cnn_dataset_dogs_cats/training_set/',
    target_size=(64, 64),  # Resize image to reduce computation required
    batch_size=32,  # default batch size
    class_mode='binary')  # cat or dog

# Preprocess test set

test_datagen = ImageDataGenerator(rescale=1. / 255)
test = test_datagen.flow_from_directory(
    '/home/zeather/data/cnn_dataset_dogs_cats/test_set/',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary')

# Model Building

cnn = tf.keras.models.Sequential()  # Initializing the CNN
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))  # Convolutional layer
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))  # Max pooling
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))  # Convolutional layer - don't need image size for subsequent layers
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))  # Max pooling
cnn.add(tf.keras.layers.Flatten())  # Flatten layer (no params required)
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))  # Fully connected ANN layer
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))  # Output layer.  Sigmoid for binary classification

# Training the CNN

cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])  # Compile the CNN
cnn.fit(x=train, validation_data=test, epochs=25)  # train the model and evaluate results on test data set

# Single Predictions

test_image = tf.keras.preprocessing.image.load_img('/home/zeather/data/cnn_dataset_dogs_cats/single_prediction/cat_or_dog_1.jpg', target_size=(64, 64))
test_image = tf.keras.preprocessing.image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)  # CNN expects a batch of images - add batch number on axis=0

pred = cnn.predict(test_image/255.0)

print(train.class_indices)  # will tell you whether cat or dog corresponds to 1 or 0

if pred[0][0] > 0.5:
    prediction = 'dog'
else:
    prediction = 'cat'

print(prediction)
