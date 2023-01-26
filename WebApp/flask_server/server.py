from flask import Flask, request
from flask_cors import CORS

import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import os 
from IA.unet.unet_model import UNet

app = Flask(__name__)
CORS(app)

model = UNet(n_channels=3, n_classes=1)

model.load_state_dict(torch.load('my_checkpoint.pth.h5')['state_dict'])
model.eval()

@app.route("/predict", methods=['POST'])
def predict():
    file = request.files['image']
    image = Image.open(file)

    width, height = image.size
    new_width = 240
    new_height = 240
    pil_image = pil_image.resize((new_width, new_height), Image.ANTIALIAS)

    to_tensor = transforms.ToTensor()
    tensor_image = to_tensor(pil_image)
    tensor_image = tensor_image.unsqueeze(0)
    try:
        preds = torch.sigmoid(model(tensor_image))
    except:
        print("error occured")

@app.route("/")
def test():
    return "test test test 111 houston everything is good..."

if __name__ == "__main__":
    app.run()

