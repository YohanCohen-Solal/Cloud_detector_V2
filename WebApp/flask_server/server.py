from flask import Flask, request
from flask_cors import CORS

from keras.models import load_model
import torch
import torchvision.transforms as transforms
import numpy as np
from keras.utils import img_to_array
from PIL import Image
from keras.applications.regnet import preprocess_input
from model_ResAttUnet import ResidualAttentionUNet

map_location=torch.device('cpu')
app = Flask(__name__)
CORS(app)

model = ResidualAttentionUNet(inputChannel=3, outputChannel=1)
torch.load('my_checkpoint.pth.h5', map_location=torch.device('cpu'))['state_dict']

#model.load_state_dict(torch.load('my_checkpoint.pth.h5')['state_dict'])
model.eval()

classification_model = load_model('res35_seg_model1.h5')
classes = ["Altocumulus", "Altostratus", "Cirrocumulus", "Cirrostratus",
           "Cirrus", "Cumulonimbus", "Cumulus", "Nimbostratus",
           "Stratocumulus", "Stratus", "clearSky"]

@app.route("/result", methods=['GET'])
def getresult(pred_class_name):
    return pred_class_name

@app.route("/predict", methods=['POST'])
def predict():
    file = request.files['image']
    image = Image.open(file)

    width, height = image.size
    new_width = 256
    new_height = 256
    pil_image = pil_image.resize((new_width, new_height), Image.ANTIALIAS)

    to_tensor = transforms.ToTensor()
    tensor_image = to_tensor(pil_image)
    tensor_image = tensor_image.unsqueeze(0)
    try:
        pred_segmentation = torch.sigmoid(model(tensor_image))
    except:
        return "Error during segmentation"

    pred_segmentation = pred_segmentation.resize((224, 224), Image.ANTIALIAS)

    x = img_to_array(pred_segmentation)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    try:
        preds = model.predict(x)
        pred_class = np.argmax(preds)
        pred_class_name = classes[pred_class]

        getresult(pred_class_name)
        return pred_class_name
    except:
        return "error during classification"


@app.route("/")
def test():
    return "test test test 111 houston everything is good..."


if __name__ == "__main__":
    app.run()
