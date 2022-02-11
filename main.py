from torchvision import models
from torchvision import transforms
import torch
from PIL import Image


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

# Import image
#img = Image.open('hase.JPG')
img = Image.open('baum_eiche.JPG')

# Pre-Process Image
img_t = transform(img)
batch_t = torch.unsqueeze(img_t, 0)

# Model Inference
alexnet.eval()

out = alexnet(batch_t)
print(out.shape)

# Read the labels/classes
with open('alexnet_labels.txt') as f:
  classes = [line.strip() for line in f.readlines()]


_, index = torch.max(out, 1)

percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100

print(classes[index[0]], percentage[index[0]].item())


_, indices = torch.sort(out, descending=True)
print([(classes[idx], percentage[idx].item()) for idx in indices[0][:5]])
