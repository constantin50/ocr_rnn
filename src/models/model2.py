import torch
from torch import nn
from torch.nn import Conv2d, MaxPool2d, BatchNorm2d, ReLU
from torchvision import models
from config import *

class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)
    def forward(self, input):
        self.rnn.flatten_parameters()
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)
        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)
        return output

class Model(nn.Module):

    def __init__(self, nHidden = 256):
        super(Model, self).__init__()
        self.resnet50 = models.resnet50(pretrained=True)
        self.resnet50.fc1 = nn.Conv2d(2048, 512, kernel_size=(2, 2))
        self.resnet50.fc2 = nn.Linear(8, 16)

        self.rnn = nn.Sequential(
            BidirectionalLSTM(nHidden*2, nHidden, nHidden),
            BidirectionalLSTM(nHidden, nHidden, len(ALPHABET)))


    def forward(self, src):
        # ResNet requires 3 channels
        if src.shape[1] == 1:
          src = src.repeat(1, 3, 1, 1)
        x = self.resnet50.conv1(src)
        x = self.resnet50.bn1(x)
        x = self.resnet50.relu(x)
        x = self.resnet50.maxpool(x)
        x = self.resnet50.layer1(x)
        x = self.resnet50.layer2(x)
        x = self.resnet50.layer3(x)
        x = self.resnet50.layer4(x)
        x = self.resnet50.fc1(x)
        b, c, h, w = x.size()
        assert h == 1, "the height of conv must be 1"
        x = x.squeeze(2) # [b, c, h*w]
        x = x.permute(2, 0, 1)  # [h*w, b, c]
        logits = self.rnn(x) # [h*w, b, num_classes]
        output = torch.nn.functional.log_softmax(logits, 2)
        return output
