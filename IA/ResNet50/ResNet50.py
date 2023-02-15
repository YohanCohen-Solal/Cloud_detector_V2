from keras.applications.resnet import ResNet50
from keras.preprocessing import image
import keras.utils as image
from keras.applications.resnet import preprocess_input, decode_predictions
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import ModelCheckpoint, EarlyStopping

# Define the number of classes
num_classes = 11

# Load the ResNet50 model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)

# Add a fully connected layer with a softmax activation
predictions = Dense(num_classes, activation='softmax')(x)

# Define the full model
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the base model's layers
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Define the preprocessing function
def preprocess_function(img):
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

# Define the data generators
train_datagen = ImageDataGenerator(preprocessing_function=preprocess_function)
val_datagen = ImageDataGenerator(preprocessing_function=preprocess_function)

checkpoint = ModelCheckpoint("resnet50_model1.h5", monitor='accuracy', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='accuracy', min_delta=0, patience=4, verbose=1, mode='auto')
callbacks_list = [checkpoint, early]

# Train the model
history = model.fit(train_datagen.flow_from_directory('DataResizeSplit/Train',
                                                      target_size=(224,224),
                                                      batch_size=32,
                                                      class_mode='categorical'),
                                                      epochs=15,
                    validation_data=val_datagen.flow_from_directory('DataResizeSplit/Test',
                                                                    target_size=(224,224),
                                                                    batch_size=32,
                                                                    class_mode='categorical'),
                    verbose=1,
                    callbacks=callbacks_list)


# Plot the accuracy and loss curves
plt.figure(figsize=[8,6])
plt.plot(history.history['accuracy'], 'r', linewidth=2.0)
plt.plot(history.history['val_accuracy'], 'b', linewidth=2.0)
plt.legend(['Training Accuracy', 'Validation Accuracy'], fontsize=18)
plt.xlabel('Epoch', fontsize=16)
plt.ylabel('Accuracy', fontsize=16)
plt.title('Accuracy Curves', fontsize=16)
plt.savefig('accuracy1_curves_res.png')
plt.show()

plt.figure(figsize=[8,6])
plt.plot(history.history['loss'], 'r', linewidth=2.0)
plt.plot(history.history['val_loss'], 'b', linewidth=2.0)
plt.legend(['Training Loss', 'Validation Loss'], fontsize=18)
plt.xlabel('Epoch', fontsize=16)
plt.ylabel('Loss', fontsize=16)
plt.title('Loss Curves', fontsize=16)
plt.savefig('loss1_curves_res.png')
plt.show()

