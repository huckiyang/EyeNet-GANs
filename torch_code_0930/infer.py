import torch.nn as nn
import torchvision.datasets
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.models as models
from collections import OrderedDict
import torch
from torch.utils.data import Dataset
import torchvision
from PIL import Image, ImageEnhance
import numpy as np
import pandas as pd
#from tqdm import *
import sys
import math
import argparse
from distutils.version import LooseVersion
import csv
import time
from sklearn.metrics import confusion_matrix
from glob import glob
from tqdm import tqdm
import csv
import os

os.makedirs('good_pics',exist_ok=True)


# from logger import Logger

#from tensorboard_logger import configure, log_value
#configure("run-1234", flush_secs=5)

# from utils import *
# from model import *
from model_zoo import model_wrapper


def to_np(x):
    return x.data.cpu().numpy()

debug = False
use_gpu = False
if(torch.cuda.is_available()):
    use_gpu = True


torch.cuda.set_device(2)
# set the logger
# if (log):
#     logger = Logger('./logs')

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

def rename_model(old_dict,new_dict):
    for i,v in old_dict.items():
        if model in i:
            new_key = i[i.find('.')+1:]
            new_dict[new_key] = v
    return new_dict
class retina_Dataset(Dataset):
    def __init__(self, root, test=True, transform=None, scale=1):
        self.root = root
        self.test = test
        self.transform = transform
        self.scale = scale
        self.data = []
        if self.test:
            imgs = glob(root+'/*.jpg')
            imgs.extend(glob(root+'/*.png',recursive=True))
            imgs.extend(glob(root+'/*.tif',recursive=True))
            imgs.extend(glob(root+'/*.jpeg',recursive=True))
            for i in range(scale):
                self.data.extend(imgs)
            print(len(self.data))
    def __getitem__(self,index):
        img_path = self.data[index]
        img = Image.open(img_path).convert('RGB')
        contrast = ImageEnhance.Contrast(img)
        img = contrast.enhance(1.5)
#         if self.transform is not None:
        img = self.transform(img)
        label = img_path
        return {'image':img, 'label':label}
    def __len__(self):
        return len(self.data)

    
batch_size = 1
    
test_set = retina_Dataset(sys.argv[1],
        transform=transforms.Compose([
            transforms.Grayscale(3),
            transforms.RandomResizedCrop(224),
            transforms.ToTensor(),
            normalize,
            ]))

test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size, shuffle=False)

resnet50 = models.resnet50()
pre = torch.load(sys.argv[2])
new = OrderedDict()
NN = model_wrapper(resnet50,3,376)

NN.load_state_dict(pre)

NN.eval()

result = []
good = []

for i,sample in tqdm(enumerate(test_loader)):
    image = Variable(sample['image'])
    predict = NN(image)
    predict = np.argmax(to_np(predict))
    print(sample['label'],predict)
    result.append(predict)
    if predict == 13:
        good.append(sample['label'])
        print(sample['label'])
count = 0

for i in result:
    if i == 13:
        count += 1
print('acc: ' ,float(count)/float(len(result)))

print(good)

with open('good_pics/'+sys.argv[1]+'csv','w+',newline='') as f:
    c = csv.writer(f)
    for i in good:
        c.writerow(i)

