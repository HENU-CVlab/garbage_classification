from math import ceil
from PIL import Image
import os
import torch
from torchvision.models import resnet50
from torchvision.transforms import transforms
import matplotlib.pyplot as plt
import numpy as np

IMAGE_DIR = "./datasets/detect_examples"
CHECKPOINT_DIR = './checkpoint/2021-12-13-11-00-resnet50-pretrained.pth'


def classify():
    transform_detect = transforms.Compose([
        transforms.Resize(256),
        # 从图像中心裁切224x224大小的图片
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])])
    classes_dict = {0: "cardboard", 1: "glass", 2: "metal", 3: "paper", 4: "plastic", 5: "trash"}
    imgs_dir = [Image.open(os.path.join(IMAGE_DIR, img)) for img in os.listdir(IMAGE_DIR)]
    imgs_ls = torch.cat([transform_detect(img).unsqueeze(0) for img in imgs_dir])
    net = resnet50(pretrained=False)
    net.load_state_dict(torch.load(CHECKPOINT_DIR))
    net.eval()
    outputs = net(imgs_ls)
    _, res = torch.max(outputs, dim=1)
    figs, axes = plt.subplots(2, ceil(len(imgs_ls) / 2), figsize=(10, 8))
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs_dir)):
        ax.imshow(np.array(img))
        ax.set_title(classes_dict[res.numpy()[i]])
    plt.show()


if __name__ == '__main__':
    classify()
