from torchvision import models
from torchvision import transforms
import torch

# list available models
dir(models)

# Load Pretrained Model (AlexNet)
alexnet = models.alexnet(pretrained=True)

# Image Transformations
transform = transforms.Compose([            #[1]
 transforms.Resize(256),                    #[2]
 transforms.CenterCrop(224),                #[3]
 transforms.ToTensor(),                     #[4]
 transforms.Normalize(                      #[5]
 mean=[0.485, 0.456, 0.406],                #[6]
 std=[0.229, 0.224, 0.225]                  #[7]
)])


