"""
implementation of architecture from 
"An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition"
https://arxiv.org/abs/1507.05717
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import models
from torch.nn import Conv2d, MaxPool2d
from config import *

class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut,dropout=0):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True,dropout=dropout)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)

        return output


class CRNN(nn.Module):

  def __init__(self,dropout=0):
    super(CRNN, self).__init__()
    self.conv0 = Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
    self.conv1 = Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
    self.conv2 = Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
    self.conv3 = Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
    self.conv4 = Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
    self.conv5 = Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
    self.conv6 = Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
    self.conv7 = Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

    self.pool1 = MaxPool2d(kernel_size=2, stride=2)
    self.pool2 = MaxPool2d(kernel_size=2, stride=(2,1))
    self.pool3 = MaxPool2d(kernel_size=2, stride=(2,1))
    self.pool4 = MaxPool2d(kernel_size=2, stride=(3,1))

    self.bn1 = nn.BatchNorm2d(64)
    self.bn2 = nn.BatchNorm2d(128)
    self.bn3 = nn.BatchNorm2d(256)
    self.bn4 = nn.BatchNorm2d(512)
    
    self.rnn = nn.Sequential(
            BidirectionalLSTM(512, nh, nh,dropout),
            BidirectionalLSTM(nh, nh, nclass,dropout))
  
  def forward(self,src):
    """
    src : [1, 3, 64, 256]
    """
    x = self.conv0(src)  # [1, 64, 64, 256]
    x = self.conv1(x)
    x = self.pool1(x)
    x = self.bn1(x)
    x = self.conv2(x)
    x = self.conv3(x)
    x = self.pool2(x)
    x = self.bn2(x)
    x = self.conv4(x)
    x = self.conv5(x)
    x = self.pool3(x)
    x = self.bn3(x)
    x = self.conv6(x)
    x = self.conv7(x)
    x = self.pool4(x) # [1, 512, 3, 125]
    x = self.bn4(x)
    x = x.flatten(2) # [1,512,3*125]
    x = x.permute(0,2,1)
    output = self.rnn(x)
    output = F.log_softmax(output, dim=2)
    #shape sequence length, batch size, input size
    output = output.permute(1,0,2)
    return output
