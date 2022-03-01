# ml_and_dl_stuff

this repo contains scripts in the context of machine learning and deep learning. The main motivation is to play around and learn about it.

## bazel/dazel
for the sake of getting to know bazel and dazel, this repo is using it. \

for example:
```
bazel run image_tagger:main -- -list_classes
```

https://docs.bazel.build/versions/main/be/python.html \
https://github.com/nadirizr/dazel \

## image tagger
This script aims at running pre-trained models (e.g. yolo5) on images and to infer detections that are stored in a sqlite database. \
The cmd line tool also allows to query for certain classes and retrieve images that contain the classes at a given confidence level. \
It is based on pytorch. \

TODO: Plan is to extend this by face detection and to train an own face recognition model based on own training data!

## alexnet
https://learnopencv.com/pytorch-for-beginners-basics/

https://learnopencv.com/pytorch-for-beginners-image-classification-using-pre-trained-models/


