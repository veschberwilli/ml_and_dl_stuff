
# from https://pytorch.org/hub/ultralytics_yolov5/

import torch
import glob
import os

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Images
# imgs = ['https://ultralytics.com/images/zidane.jpg']  # batch of images
imgs = glob.glob(os.path.join('pics', '*.jpg'))

# Inference
results = model(imgs)

# Results
results.print()
results.save()  # or .show()

#results.xyxy[0]  # img1 predictions (tensor)
tmp = results.pandas().xyxy[0]  # img1 predictions (pandas)
print(tmp)

target_class = f"horse"
idx = 0
for pic in tmp:
    if target_class in pic['name'].values:
        print(f"{target_class} found in {imgs[idx]}")
    idx += 1
