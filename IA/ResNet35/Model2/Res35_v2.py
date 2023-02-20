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
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# Define the number of classes
num_classes = 11

# Split dataset
import os
import random
import shutil

def __copyfiles__(_folder, dataset_src,_class_, train_or_test_folder):
    for _filename in _folder:
            src_path = os.path.join(dataset_src,_class_, _filename)
            dst_path = os.path.join(train_or_test_folder,_class_, _filename)
            shutil.copy2(src_path, dst_path)
def create_folder_and_class(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
            
def split_data(src_folder, split_ratio):
    print('\n\n\n','First we have to create a new dataset which will contain the datas of train (80%) and test(20%) separated randomly')
    new_dataset=input('Give a name for the new dataset : ')
    train_folder=os.path.join(new_dataset,'Train')
    test_folder=os.path.join(new_dataset,'Test')
    create_folder_and_class(train_folder)
    create_folder_and_class(test_folder)

    src_folder_dir=os.listdir(src_folder)
    for _class_ in src_folder_dir:
        create_folder_and_class(os.path.join(train_folder,_class_))
        create_folder_and_class(os.path.join(test_folder,_class_))
        filenames = os.listdir(os.path.join(src_folder,_class_))
        random.shuffle(filenames)
        num_train = int(split_ratio * len(filenames))
        train_filenames = filenames[:num_train]
        test_filenames = filenames[num_train:]
        __copyfiles__(train_filenames, src_folder,_class_, train_folder)
        __copyfiles__(test_filenames, src_folder,_class_, test_folder)
    
    return train_folder, test_folder

# Load the ResNet50 model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the base model's layers
for layer in base_model.layers:
    layer.trainable = False

# Add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dense(512, activation='relu')(x)

# Add a fully connected layer with a softmax activation
predictions = Dense(num_classes, activation='softmax')(x)

# Define the full model
model = Model(inputs=base_model.input, outputs=predictions)



# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Define the preprocessing function
def preprocess_function(img):
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

# Define the data generators
train_datagen = ImageDataGenerator(preprocessing_function=preprocess_function)
val_datagen = ImageDataGenerator(preprocessing_function=preprocess_function)

checkpoint = ModelCheckpoint("resnet35_model2.h5", monitor='accuracy', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', save_freq=1)
early = EarlyStopping(monitor='accuracy', min_delta=0, patience=4, verbose=1, mode='auto')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
callbacks_list = [checkpoint, early, reduce_lr]

train_folder, test_folder = split_data('DataResize', 0.8)

X_train=train_datagen.flow_from_directory(train_folder,
                                          target_size=(224,224),
                                          batch_size=32,
                                          class_mode='categorical')
X_test=val_datagen.flow_from_directory(test_folder,
                                       target_size=(224,224),
                                       batch_size=32,
                                       class_mode='categorical')
# Train the model
history = model.fit(X_train,epochs=20,
                    validation_data=X_test,
                    verbose=1,
                    callbacks=callbacks_list)

acc=history.history['accuracy']
val_acc=history.history['val_accuracy']
loss=history.history['loss']
val_loss=history.history['val_loss']

# Plot the accuracy and loss curves
plt.figure(figsize=[8,6])
plt.plot(acc, 'r', linewidth=2.0)
plt.plot(val_acc, 'b', linewidth=2.0)
plt.legend(['Training Accuracy', 'Validation Accuracy'], fontsize=18)
plt.xlabel('Epoch', fontsize=16)
plt.ylabel('Accuracy', fontsize=16)
plt.title('Accuracy Curves', fontsize=16)
plt.savefig('accuracy2_res35.png')


plt.figure(figsize=[8,6])
plt.plot(loss, 'r', linewidth=2.0)
plt.plot(val_loss, 'b', linewidth=2.0)
plt.legend(['Training Loss', 'Validation Loss'], fontsize=18)
plt.xlabel('Epoch', fontsize=16)
plt.ylabel('Loss', fontsize=16)
plt.title('Loss Curves', fontsize=16)
plt.savefig('loss2_res35.png')