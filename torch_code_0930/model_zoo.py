# Freddy @Mann Coffee, BUPT
# April 30, 2018

# reference: http://pytorch.org/docs/master/torchvision/models.html

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
import numpy as np



class model_wrapper(nn.Module):
    def __init__(self, model, channels_in=1,channels_out=2):
        super(model_wrapper, self).__init__()
        self.model = model
#         self.one2three = nn.Sequential(
#                 nn.Conv2d(channels_in,3,kernel_size=1,padding=0))
        self.linear = nn.Sequential(
                nn.Linear(1000,channels_out))
    def forward(self, x):
        # out = self.one2three(x)
        out = self.model(x)
        out = self.linear(out)
        return out

if __name__ == "__main__":
    """
    testing
    """
    p = True
    vgg16 = models.vgg16(pretrained=p)
    resnet18 = models.resnet18(pretrained=p)
    densenet = models.densenet161(pretrained=p)
    inception = models.inception_v3(pretrained=p)

    test = Variable(torch.FloatTensor(np.random.random((1,1,224,224))))
    wrapped_model = model_wrapper(vgg16,1,2)
    out = wrapped_model(test)
    print(out.shape)
    loss1 = torch.sum(out)
    loss1.backward()
