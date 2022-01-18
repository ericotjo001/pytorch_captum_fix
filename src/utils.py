import numpy as np
import torch
from torchvision import transforms

import matplotlib.pyplot as plt
from PIL import Image
from skimage.transform import resize

def tonumpy(x):
    # x is torch tensor
    return x.clone().detach().cpu().numpy()

def show_attribute(pil_img,x, heatmaps, h,w ,title=None):
    plt.figure(figsize=(12,5))
    plt.gcf().add_subplot(141)
    plt.gca().imshow(pil_img)
    plt.gcf().add_subplot(142)
    plt.gca().imshow(pil_img)
    plt.gca().imshow(resize(heatmaps[0,:,:], (h,w)), alpha=0.5)
    plt.gca().set_xlabel('target: 243 # correct label for dog')
    if title is not None:
        plt.title(title)

    plt.gcf().add_subplot(143)
    plt.gca().imshow(pil_img)
    plt.gca().imshow(resize(heatmaps[1,:,:], (h,w)), alpha=0.5)
    plt.gca().set_xlabel('target: 281 # correct tabby cat')
    plt.gcf().add_subplot(144)
    plt.gca().imshow(tonumpy(x[2,:,:,:]).transpose(1,2,0))
    plt.gca().imshow(resize(heatmaps[2,:,:], (h,w)), alpha=0.5)



    plt.show()
    plt.tight_layout()
    plt.close()


def show_attribute2(pil_img,x, heatmaps, h,w ,title=None):
    plt.figure(figsize=(12,5))
    plt.gcf().add_subplot(141)
    plt.gca().imshow(pil_img)
    plt.gcf().add_subplot(142)
    plt.gca().imshow(resize(heatmaps[0,:,:], (h,w)), vmin=-1,vmax=1, cmap='bwr')
    plt.gca().imshow(pil_img, alpha=0.1)
    plt.gca().set_xlabel('target: 243 # correct label for dog')

    if title is not None:
        plt.title(title)

    plt.gcf().add_subplot(143)
    plt.gca().imshow(resize(heatmaps[1,:,:], (h,w)),  vmin=-1,vmax=1, cmap='bwr')
    plt.gca().imshow(pil_img, alpha=0.1)
    plt.gca().set_xlabel('target: 281 # correct tabby cat')
    plt.gcf().add_subplot(144)
    plt.gca().imshow(resize(heatmaps[2,:,:], (h,w)), vmin=-1,vmax=1, cmap='bwr')
    plt.gca().imshow(tonumpy(x[2,:,:,:]).transpose(1,2,0), alpha=0.1)



    plt.show()
    plt.tight_layout()
    plt.close()



def get_image(device=None):
    """
    pil_img: PIL image of a dog and a cat
    x      : batch of 3 images (dog/cat img, dog/cat img, noise)
    """
    batch_size = 3
    img_dir = 'cat_dog.png'
    pil_img = Image.open(img_dir)
    h,w,c = np.asarray(pil_img).shape
    img = torch.from_numpy(np.asarray(pil_img).transpose(2,0,1))
    img= img.unsqueeze(0).to(torch.float)

    normalizeTransform = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225])
    normalizeImageTransform = transforms.Compose([transforms.ToTensor(), 
        normalizeTransform])

    x = torch.randn(size=(batch_size,3,h,w)).to(torch.float)
    x[0,:,:,:] = x[0,:,:,:]*0 + img
    x[1,:,:,:] = x[1,:,:,:]*0 + img
    x[2,:,:,:] = torch.clip(x[2,:,:,:],0,1.)
    labels = torch.randint(0,1000, size=(batch_size,)).to(torch.long)
    labels[0] = 243 # correct label for dog
    labels[1] = 281 # tabby cat

    x = x.to(device=device)
    labels = labels.to(device=device)

    return pil_img, x, labels, normalizeTransform


def process_color_heatmaps(heatmaps, normalize='[0,1]'):
    # heatmaps: torch tensor shape (b,c,h,w)
    b,c,h,w = heatmaps.shape

    with torch.no_grad():
        heatmaps = torch.sum(heatmaps,axis=1)

    heatmaps = tonumpy(heatmaps)

    if normalize=='[0,1]':
        for i in range(b):
            heatmaps[i] = heatmaps[i] - np.min(heatmaps[i].reshape(-1))
            heatmaps[i] = heatmaps[i]/np.max(np.abs(heatmaps[i].reshape(-1)))
    elif normalize=='[-1,1]':
        for i in range(b):
            heatmaps[i] = heatmaps[i] / np.max(np.abs(heatmaps[i].reshape(-1)))
    else:
        raise NotImplementedError()

    return heatmaps

def normalize_cam_heatmaps(heatmaps):
    # heatmaps shape must be b,h,w tensor
    b,h,w = heatmaps.shape
    heatmaps = tonumpy(heatmaps)
    hmap = np.zeros(shape=(b,h,w)) + heatmaps

    for i in range(b):
        hmap[i] = resize(hmap[i], (h,w))
        hmap[i] = hmap[i] - np.min(hmap[i].reshape(-1))
        hmap[i] = hmap[i]/np.max(np.abs(hmap[i].reshape(-1)))
    return hmap