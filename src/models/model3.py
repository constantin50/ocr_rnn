import torch
from torch import nn
from torch.nn import Conv2d, MaxPool2d, BatchNorm2d, ReLU
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

class CRNN(nn.Module):

    def __init__(self, nHidden = 256):
        super(CRNN, self).__init__()

        self.conv0 = Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.conv1 = Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv5 = Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv6 = Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv7 = Conv2d(512, 512, kernel_size=7, stride=1, padding=1)

        self.pool1 = MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = MaxPool2d(kernel_size=2, stride=(2,1))
        self.pool3 = MaxPool2d(kernel_size=2, stride=(2,1))
        self.pool4 = MaxPool2d(kernel_size=2, stride=(3,1))

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)

        self.rnn = nn.Sequential(
            BidirectionalLSTM(nHidden*2, nHidden, nHidden),
            BidirectionalLSTM(nHidden, nHidden, len(ALPHABET)))


    def forward(self, src):
        
        x = self.conv0(src)
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
        #print(x.shape)
        b, c, h, w = x.size()
        assert h == 1, "the height of conv must be 1"
        x = x.squeeze(2) # [b, c, h*w]
        x = x.permute(2, 0, 1)  # [h*w, b, c]
        logits = self.rnn(x) # [h*w, b, num_classes]
        output = torch.nn.functional.log_softmax(logits, 2)
        return output
