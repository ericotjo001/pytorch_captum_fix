# Editing pretrained model for Pytorch CAPTUM

Problem: pytorch captum sometimes is not compatible with the shape of the architecture. You might get something like "A Module ReLU(inplace=True) was detected that does not contain some of the input/output attributes that are required for DeepLift computations" when using DeepLIFT etc.

For now, this repository contains:
1. Fixes to DeepLIFT problem for Resnet34 and Resnet50
2. Class Activation Mapping (CAM) for comparison
 
Usage: simply run the python file in the folder. eg. python resnet50_deeplift.py

<img src="https://drive.google.com/uc?export=view?&id=1Bk8yN_gicm_UmBwbIB6bDDxVaVvQmkfi" width="640"></img>

