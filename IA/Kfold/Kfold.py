import os
import numpy as np
from sklearn.model_selection import KFold
from keras.applications.resnet import ResNet50
from keras.preprocessing import image
import keras.utils as image
from keras.applications.resnet import preprocess_input, decode_predictions
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# Define the number of folds
k = 5

# Define the root directory of the image dataset
root_dir = "path/to/dataset"

# Define the ImageDataGenerator for data augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

# Get the list of subdirectories in the root directory
classes = os.listdir(root_dir)

# Create a list of all image filenames with their corresponding class label
filenames = []
for class_name in classes:
    class_dir = os.path.join(root_dir, class_name)
    for filename in os.listdir(class_dir):
        filenames.append((os.path.join(class_dir, filename), class_name))

# Convert the list of filenames to a numpy array
filenames = np.array(filenames)

# Shuffle the array of filenames
np.random.shuffle(filenames)

# Use KFold to split the data into k folds
kf = KFold(n_splits=k)

# Loop over each fold
for fold_idx, (train_indices, val_indices) in enumerate(kf.split(filenames)):
    print(f"Fold {fold_idx+1}:")
    
    # Split the filenames into training and validation sets for this fold
    train_filenames = filenames[train_indices]
    val_filenames = filenames[val_indices]
    
    # Define the ImageDataGenerator for the training set (including data augmentation)
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True
    )
    
    # Define the ImageDataGenerator for the validation set (no data augmentation)
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    # Define the generators for the training and validation sets
    train_generator = train_datagen.flow_from_directory(
        directory=root_dir,
        classes=classes,
        class_mode="categorical",
        batch_size=32,
        subset="training",
        shuffle=True
    )
    
    val_generator = val_datagen.flow_from_directory(
        directory=root_dir,
        classes=classes,
        class_mode="categorical",
        batch_size=32,
        subset="validation",
        shuffle=True
    )
    
    # Train the model on the training set and evaluate it on the validation set
    # Replace this with your own code for training and evaluating the model
    model.fit(
        train_generator,
        epochs=10,
        validation_data=val_generator
    )
