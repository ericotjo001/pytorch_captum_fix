from src.adjusted_model_component import BasicBlockAdjusted
from src.utils import tonumpy, get_image, show_attribute, show_attribute2, process_color_heatmaps

import torch
import torch.nn as nn
import torchvision.models as mod


import numpy as np

class Resnet50c(nn.Module):
    def __init__(self):
        super(Resnet50c, self).__init__()
    
        self.backbone = mod.resnet34(pretrained=True, progress=False)

        self.adjust_for_captum_problem()
        # self.print_after_captum_adjustment() # or debugging

    def verify_equality(self,x):
        print('verify_equality...')
        with torch.no_grad():
            y1 = self.forward_core_model(x)
            y = self.backbone(x)
        print(y[0,:4])
        print(y1[0,:4])
        assert(np.all(tonumpy(y)==tonumpy(y1)))
        print('equality ok!')

    def forward_core_model(self,x):
        """ core module 'backbone' has the following named children:
            0  conv1
            1  bn1
            2  relu
            3  maxpool
            4  layer1
            5  layer2
            6  layer3
            7  layer4
            8  avgpool
            9  fc  
        """
        y1 = x.clone().detach()
        
        for i,(mkey,_) in enumerate(self.backbone.named_children()):
            # print('%-2s'%(str(i)), mkey)
            if mkey=='fc':
                y1 = y1.squeeze(3).squeeze(2)
            y1 = self.backbone.__getattr__(mkey)(y1)
        return y1
        
    def forward(self,x):
        # do any modification here if necessary
        # e.g. if you want to find tune using different FC layer, you can do 
        #    forward propagation selectively
        x = self.forward_core_model(x)
        return x

    def adjust_for_captum_problem(self):
        setattr(self.backbone,'relu',nn.ReLU() )
        for i,(layer_name,m) in enumerate(self.backbone.named_children()):    
            if type(m) == nn.Sequential:
                for j,(sublayer_name,m2) in enumerate(m.named_children()):
                    if type(m2) == mod.resnet.BasicBlock:
                        # setattr(getattr(getattr(self.backbone, layer_name), sublayer_name),'relu', nn.ReLU())
                        temp = BasicBlockAdjusted()
                        temp.inherit_weights(m2)
                        setattr(getattr(self.backbone, layer_name), sublayer_name, temp)

    def print_after_captum_adjustment(self):
        for i,(layer_name,m) in enumerate(self.backbone.named_children()):
            print(type(m))
            for j,(sublayer_name,m2) in enumerate(m.named_children()):
                print(type(m2))
                for k,(subsublayer_name,m3) in enumerate(m2.named_children()):
                    print(type(m3))


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = Resnet50c().to(device=device)

    """
    pil_img: PIL image of a dog and a cat
    x      : batch of 3 images (dog/cat img, dog/cat img, noise)
    """
    pil_img, x, labels, normalizeTransform = get_image(device=device)

    b,c,h,w = x.shape
    x.requires_grad=True

    with torch.no_grad():
        net.verify_equality(x)
    y = net(normalizeTransform(x))

    from captum.attr import DeepLift
    heatmaps = DeepLift(net.backbone).attribute(normalizeTransform(x), target=labels)
    heatmaps = process_color_heatmaps(heatmaps, normalize='[-1,1]')
    print(heatmaps.shape, np.max(heatmaps), np.min(heatmaps))

    show_attribute2(pil_img,x, heatmaps, h,w, title='resnet 50 + DeepLIFT' )