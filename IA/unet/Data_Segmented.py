import torch
import torchvision
import torchvision.transforms as transforms
import os 
from model_ResAttUnet import ResidualAttentionUNet
from PIL import Image

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Disable gradient computation during inference
torch.set_grad_enabled(False)

# Initialize model and load saved parameters
model = ResidualAttentionUNet(inputChannel=3, outputChannel=1)
model.load_state_dict(torch.load('my_checkpoint.pth.h5')['state_dict'])
model.to(device)
model.eval()

# Specify input and output directories
input_dir = "IA/Data"
output_dir = "IA/DataSegmented"

# Loop over subdirectories in input directory
for folder in os.listdir(input_dir):
    input_folder = os.path.join(input_dir, folder)
    output_folder = os.path.join(output_dir, folder)
    
    # Create output directory if it does not exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Loop over images in input directory
    for image_filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, image_filename)
        
        # Load and preprocess image
        pil_image = Image.open(input_path).resize((256, 256), Image.ANTIALIAS)
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        tensor_image = transform(pil_image).unsqueeze(0).to(device)
        
        # Run inference and save output
        with torch.no_grad():
            preds = torch.sigmoid(model(tensor_image))
        output_path = os.path.join(output_folder, image_filename)
        torchvision.utils.save_image(preds, output_path)