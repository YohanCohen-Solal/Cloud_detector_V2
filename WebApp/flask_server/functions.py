import os
import tensorflow as tf
import numpy as np

from keras.models import load_model
import torch
import torchvision.transforms as transforms
import numpy as np
from keras.utils import img_to_array
from PIL import Image
from keras.applications.regnet import preprocess_input
from model_ResAttUnet import ResidualAttentionUNet

model = ResidualAttentionUNet(inputChannel=3, outputChannel=1)
model.load_state_dict(torch.load('my_checkpoint.pth.h5')['state_dict'])
model.eval()

classification_model = load_model('res35_seg_model1.h5')
classes = ["Altocumulus", "Altostratus", "Cirrocumulus", "Cirrostratus",
           "Cirrus", "Cumulonimbus", "Cumulus", "Nimbostratus",
           "Stratocumulus", "Stratus", "clearSky"]

classification_no_segmentaiton = load_model('resnet_model2.h5')

def preprocess_image(step, images):
    if step == "segmentation":
        print("je suis passer par le preprocess segmentation")
        print("j'ai ouvert l'image")
        width, height = images.size
        new_width = 256
        new_height = 256
        pil_image = images.resize((new_width, new_height), Image.ANTIALIAS)
        print("j'ai réussi à resize l'image")
        to_tensor = transforms.ToTensor()
        tensor_image = to_tensor(pil_image)
        tensor_image = tensor_image.unsqueeze(0)
        print("je vais retourner l'image sous forme de tenseur...")

        return tensor_image
    
    elif step == "classification":
        print("je suis passer par le preprocess classification")
        print("j'ai réussi à resize l'image")

        x = img_to_array(images)
        print("j'ai réussi à convertir l'image en array")
        x = np.expand_dims(x, axis=0)
        print("j'ai réussi à augmenter la dimention")
        x = preprocess_input(x)
        print("j'ai réussi à preprocess l'imput")
        print("je vais maintenant retourner l'image preprocess pour la classification")

        return x


def predict(step, image):
    print(step +": ")
    if step == "segmentation":
        print("je suis entré dans la fonction predict pour la segmentation")
        img = preprocess_image(step, image)
        pred_segmentation = torch.sigmoid(model(img))
        return pred_segmentation
    
    elif step == "classification":
        print("je suis entré dans la fonction predict pour la classification")
        img = preprocess_image(step, image)
        preds = classification_model.predict(img)
        pred_class = np.argmax(preds)
        pred_class_name = classes[pred_class]
        print("je vais maintenant retourner la classe")

        return pred_class_name

def classificatieur(images):
        print("je suis passer par le preprocess classification")
        print("j'ai réussi à resize l'image")

        x = img_to_array(images)
        print("j'ai réussi à convertir l'image en array")
        x = np.expand_dims(x, axis=0)
        print("j'ai réussi à augmenter la dimention")
        x = preprocess_input(x)
        print("j'ai réussi à preprocess l'imput")
        print("je vais maintenant retourner l'image preprocess pour la classification")

        print("je suis entré dans la fonction predict pour la classification")
        preds = classification_model.predict(x)
        pred_class = np.argmax(preds)
        pred_class_name = classes[pred_class]
        print("je vais maintenant retourner la classe")

        return pred_class_name