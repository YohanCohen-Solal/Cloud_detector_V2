import numpy as np
#from keras.preprocessing import image
import keras.utils as image
from keras.applications.resnet import preprocess_input, decode_predictions
from keras.models import load_model

# Load the pre-trained ResNet50 model
model = load_model('resnet35_model2.h5')

# Define the classes of your task
classes = ["Altocumulus", "Altostratus", "Cirrocumulus", "Cirrostratus", "Cirrus", "clearSky", "Cumulonimbus", "Cumulus", "Nimbostratus", "Stratocumulus", "Stratus"]

# Load the image for prediction and resize it to 224x224 pixels
img_path = 'DALL_E_2023-02-24_13.08.36-Cumulonimbus.png'
img = image.load_img(img_path, target_size=(224, 224))

# Convert the image to a numpy array and preprocess it
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# Make the prediction and print the result
preds = model.predict(x)
pred_class = np.argmax(preds)
pred_class_name = classes[pred_class]
print('Predicted class:', pred_class_name)
