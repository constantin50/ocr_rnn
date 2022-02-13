"""
"Fine-tuning Handwriting Recognition systems with Temporal Dropout" https://arxiv.org/pdf/2102.00511v1.pdf
"""
import random
import torch
from torch import nn
from torch.nn import Conv2d, MaxPool2d, BatchNorm2d, LeakyReLU
from config import *

def TemporalDropout(x, p = 0.2):
    B, C, WH = x.shape # BATCH_SIZE, CHANNELS, WIDTH*HEIGHT
    v = torch.ones(size=(WH,))
    for k in range(int(C*p)):
      i = random.randint(0,WH-1)
      v[i] = 0
    mask = torch.stack([v]*C).to(DEVICE)
    x = x*mask
    return x

class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut, num_layers):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True, num_layers = num_layers)
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

    def __init__(self, nChannels, nHidden, num_classes, dropout = 0.2):
        super(Model, self).__init__()
        
        self.dropout = dropout
        
        self.act = LeakyReLU(negative_slope=0.01, inplace=False)
        self.conv0 = Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.conv1 = Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv5 = Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv6 = Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv7 = Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv8 = Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv9 = Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv10 = Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv11 = Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv12 = Conv2d(512, 512, kernel_size=4, stride=1, padding=1)

        self.pool1 = MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = MaxPool2d(kernel_size=2, stride=2)
        self.pool3 = MaxPool2d(kernel_size=(2,1), stride=(2,1))
        self.pool4 = MaxPool2d(kernel_size=(2,1), stride=(2,1))
        self.pool5 = MaxPool2d(kernel_size=(2,1), stride=(2,1))

        self.bn1 = BatchNorm2d(64)
        self.bn2 = BatchNorm2d(128)
        self.bn3 = BatchNorm2d(256)
        self.bn4 = BatchNorm2d(512)

        self.rnn1 = BidirectionalLSTM(2*nHidden, nHidden, num_classes, num_layers=3)
        self.rnn2 = BidirectionalLSTM(2*nHidden, nHidden, num_classes, num_layers=3)


    def forward(self, src):
        
        x = self.act(self.bn1(self.conv0(src)))
        x = self.act(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.act(self.bn2(self.conv2(x)))
        x = self.act(self.bn2(self.conv3(x)))
        x = self.pool2(x)
        x = self.act(self.bn3(self.conv4(x)))
        x = self.act(self.bn3(self.conv5(x)))
        x = self.act(self.bn3(self.conv6(x)))
        x = self.pool3(x)
        x = self.act(self.bn4(self.conv7(x)))
        x = self.act(self.bn4(self.conv8(x)))
        x = self.act(self.bn4(self.conv9(x)))
        x = self.pool4(x)
        x = self.act(self.bn4(self.conv10(x)))
        x = self.act(self.bn4(self.conv11(x)))
        x = self.act(self.bn4(self.conv12(x)))
        x = self.pool5(x)
        b, c, h, w = x.size()
        assert h == 1, "the height of conv must be 1"
        x = x.squeeze(2) # [b, c, h*w]
        x = TemporalDropout(x, self.dropout)
        x = x.permute(2, 0, 1)  # [h*w, b, c]
        output1 = self.rnn1(x)
        output2 = self.rnn2(x)
        output = torch.cat([output1, output2], 0)
        output = torch.nn.functional.log_softmax(output, 2)
        return output
