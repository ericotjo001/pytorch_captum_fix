import torch
import torch.nn as nn
import torchvision.models as mod
from torchvision import transforms

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.transform import resize

from src.utils import tonumpy, get_image, show_attribute, normalize_cam_heatmaps

def tonumpy(x):
    return x.clone().detach().cpu().numpy()

class ResNet50(nn.Module):
    def __init__(self,):
        super(ResNet50, self).__init__()
        self.backbone = mod.resnet50(pretrained=True, progress=False)

    def forward(self,x, ):
        return self.backbone(x)

    def verify_equality(self,x):
        print('verify_equality...')
        with torch.no_grad():
            y1 = x.clone().detach()
            
            for mkey,_ in self.backbone.named_children():
                if mkey=='fc':
                    y1 = y1.squeeze(3).squeeze(2)
                y1 = self.backbone.__getattr__(mkey)(y1)

            y = self.backbone(x)
        print(y[0,:4])
        print(y1[0,:4])
        assert(np.all(tonumpy(y)==tonumpy(y1)))
        print('equality ok!')
        

    def compute_cam(self,x, labels):
        # we borrow this from https://github.com/clovaai/wsolevaluation
        # x = self.conv1(x)
        # x = self.bn1(x)
        # x = self.relu(x)
        # x = self.maxpool(x)

        # x = self.layer1(x)
        # x = self.layer2(x)
        # x = self.layer3(x)
        # x = self.layer4(x)

        with torch.no_grad():
            for mkey,_ in self.backbone.named_children():
                x = self.backbone.__getattr__(mkey)(x)
                if mkey =='layer4':
                    break
        feature_map = x.detach().clone()
        cam_weights = self.backbone.__getattr__('fc').weight[labels]
        cams = (cam_weights.view(*feature_map.shape[:2], 1, 1) *
                feature_map).mean(1, keepdim=False)
        return cams


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = ResNet50().to(device=device)

    """
    pil_img: PIL image of a dog and a cat
    x      : batch of 3 images (dog/cat img, dog/cat img, noise)
    """
    pil_img, x, labels, normalizeTransform = get_image(device=device)
    b,c,h,w = x.shape

    with torch.no_grad():
        net.verify_equality(x)
    y = net(normalizeTransform(x))
    heatmaps = net.compute_cam(x, labels)

    heatmaps = normalize_cam_heatmaps(heatmaps)

    show_attribute(pil_img,x, heatmaps, h,w , title='resnet 50 + CAM')