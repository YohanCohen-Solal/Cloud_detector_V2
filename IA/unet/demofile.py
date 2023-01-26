import torch
import torchvision
import torchvision.transforms as transforms
import os 

from unet_model import UNet
from Classification import ConvNet
from PIL import Image


model_segmentation = UNet(n_channels=3, n_classes=1)
model_segmentation.load_state_dict(torch.load('my_checkpoint.pth.h5')['state_dict'])

model_classification=ConvNet(num_classes=11)
model_classification.load_state_dict(torch.load('best_checkpoint.h5'))


img_path = "IA/Data/altocumulus/image14.jpeg"
pil_image = Image.open(img_path)

width, height = pil_image.size
new_width = 240
new_height = 240
pil_image = pil_image.resize((new_width, new_height), Image.ANTIALIAS)
to_tensor = transforms.ToTensor()

tensor_image = to_tensor(pil_image)
tensor_image = tensor_image.unsqueeze(0)
preds = torch.sigmoid(model_segmentation(tensor_image))
preds = torch.cat((preds, preds, preds), dim=1)
output = model_classification(preds)

prediction = output.argmax()