import os
import sys

from torch.autograd import Variable

import torchattacks
from demo.utils import imshow, get_pred

# sys.path.insert(0, '..')
import robustbench
from robustbench.data import load_cifar10, load_imagenetc
from robustbench.utils import load_model, clean_accuracy
from torchvision import models
import numpy as np
import matplotlib.pyplot as plt
import json

import torch
import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import PIL


def image_folder_custom_label(root, transform, idx2label, idex2name):
    old_data = dsets.ImageFolder(root=root, transform=transform)
    old_classes = old_data.classes

    label2idx = {}

    for i, item in enumerate(idx2label):
        label2idx[item] = i

    new_data = dsets.ImageFolder(root=root, transform=transform,
                                 target_transform=lambda x: idex2name.index(old_classes[x]))
    new_data.targets = [idex2name.index(new_data.classes[k]) for k in range(len(new_data.classes))]
    new_data.classes = idx2label
    new_data.class_to_idx = label2idx

    return new_data


def get_imagenet_data(n_examples):
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    # https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json
    class_idx = json.load(open("./data/imagenet_class_index.json"))
    idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]
    idx2name = [class_idx[str(k)][0] for k in range(len(class_idx))]
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor()
    ])

    imagnet_data = image_folder_custom_label(root='./data/val',
                                             transform=transform,
                                             idx2label=idx2label,
                                             idex2name=idx2name)
    data_loader = torch.utils.data.DataLoader(imagnet_data, batch_size=1, shuffle=False)
    x_test, y_test = [], []
    for step, (x, y) in enumerate(data_loader):
        x_test.append(x)
        y_test.append(y)
        if n_examples is not None and n_examples - 1 <= step:
            break
    x_test_tensor = torch.cat(x_test)
    y_test_tensor = torch.cat(y_test)
    return x_test_tensor, y_test_tensor


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with PIL.Image.open(f) as img:
            return img.convert('RGB')


model = models.resnet18(pretrained=True).to('cpu').eval()
class_idx = json.load(open("./data/imagenet_class_index.json"))
idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]

images, labels = get_imagenet_data(2)
# print(labels)
print(images[0])

device = "cpu"
idx = 1

# atk = torchattacks.CW(model, steps=10)
# atk = torchattacks.DeepFool(model, steps=10)
# atk = torchattacks.JSMA(model)
atk = torchattacks.PGD(model, eps=16 / 255, alpha=2 / 225, steps=10, random_start=True, targeted=False)
# atk = torchattacks.APGD(model, eps=16 / 255, steps=10)
# atk = torchattacks.PGDL2(model, eps=1000 / 255, alpha=2 / 225, steps=10, random_start=True)
# atk = torchattacks.FGSM(model, eps=13 / 255)
# atk = torchattacks.JSMA(model)

# atk.set_mode_targeted_by_label(quiet=True)
# target_labels = (labels + 3) % 10

# pdg_image = atk(images, target_labels)

pdg_image = atk(images, labels)
# pdg_pre = get_pred(model, pdg_image[idx:idx + 1], device)
# print("attack: original label", idx2label[labels[idx:idx + 1].item()], " pred label", idx2label[pdg_pre.item()])

# print("attack: original label", idx2label[labels[idx:idx + 1].item()], " pred label", idx2label[pdg_pre.item()])

# imshow(pdg_image[idx:idx + 1], title="1")
imshow(pdg_image[idx:idx + 1],
       title="True:%s, Now:%s" % (idx2label[labels[idx:idx + 1].item()], idx2label[pdg_pre.item()]))

