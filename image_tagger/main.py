"""
Flow:

1. load the model
2. get images
3. transform images
4. run inference
5. create exif tags

Another Tool: tag_finder
tool to query exif tags

Stretch Goal: train own face recognition model
"""
# from https://pytorch.org/hub/ultralytics_yolov5/

import torch
import glob
import os
import logging

logging.basicConfig(level=logging.DEBUG)

# params
model_name = 'yolov5s'
model_source = 'ultralytics/yolov5'
path_to_imgs = os.path.join('..', '..', 'pics')
supported_img_formats = ['jpg']

# start
logging.info(f"Start Image Tagger...")

# load model
logging.info(f"get the model {model_name}")
model = torch.hub.load(model_source, model_name, pretrained=True)

# get images
logging.debug(f"Base Dir: {os.getcwd()}")
logging.info(f"Locking for images in {path_to_imgs}")
# TODO: supported_img_formats
imgs = glob.glob(os.path.join(path_to_imgs, '*.jpg'))

# proceed if images found
if len(imgs) == 0:
    raise Exception(f"No Images Found Under {path_to_imgs}.")

# inference
results = model(imgs)

# Results
results.print()
results.save()  # or .show()

#results.xyxy[0]  # img1 predictions (tensor)
tmp = results.pandas().xyxy[0]  # img1 predictions (pandas)
logging.info(tmp)

target_class = f"horse"
idx = 0
for pic in tmp:
    if target_class in pic['name'].values:
        logging.info(f"{target_class} found in {imgs[idx]}")
    idx += 1
