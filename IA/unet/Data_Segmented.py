import torch
import torchvision
import numpy as np
import albumentations as A
import torchvision.transforms as transforms
import os 

from unet_model import UNet
from PIL import Image
# Model class must be defined somewhere

model = UNet(n_channels=3, n_classes=1)

model.load_state_dict(torch.load('my_checkpoint.pth.h5')['state_dict'])
model.eval()

directory = "IA/Data"
for folder in os.listdir(directory):
    f = os.path.join(directory, folder)
    isExist = os.path.exists(f"IA/DataSegmented/{folder}")
    if isExist == False:
        os.mkdir(f"IA/DataSegmented/{folder}")
    for images in os.listdir(f):

        print(images)
        img_path = os.path.join(f,images)
        pil_image = Image.open(img_path)

        width, height = pil_image.size
        new_width = 240
        new_height = 240
        pil_image = pil_image.resize((new_width, new_height), Image.ANTIALIAS)

        to_tensor = transforms.ToTensor()
        tensor_image = to_tensor(pil_image)
        tensor_image = tensor_image.unsqueeze(0)
        try:
            preds = torch.sigmoid(model(tensor_image))
            torchvision.utils.save_image(preds, f"IA/DataSegmented/{folder}/{images}")
        except:
            print("error occured")
        