import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import models

class BidirectionalLSTM(nn.Module):
    def __init__(self, nIn, nHidden, nOut,dropout=0):
        """
        params
        ---
        nIn : int
          input dimension
        nHidden : int
          hidden dimension
        nOut : int
          output dimension
        dropout : float
        """
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True,dropout=dropout)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        """
        params
        ---
        input : torch.tensor
          shape (L,B,nHidden), L - length, B - batch size
        
        returns
        ---
        output : torch.tensor
          
        """
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)

        return output

class ResNetRNN(nn.Module):
    def __init__(self,dropout=0):
        super(ResNetRNN, self).__init__()
        self.resnet50 = models.resnet50(pretrained=True)
        self.resnet50.fc1 = nn.Conv2d(2048, int(512/2), 1)
        self.resnet50.fc2 = nn.Linear(8, 16)
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, nh, nh,dropout),
            BidirectionalLSTM(nh, nh, nclass,dropout))


    def forward(self,src):
        '''
        params
        ---
        src : torch.tensor [64, 3, 64, 256] : [B,C,H,W]
            B - batch, C - channel, H - height, W - width
        
        output : torch.tensor
        '''
        x = self.resnet50.conv1(src)
        x = self.resnet50.bn1(x)
        x = self.resnet50.relu(x)
        x = self.resnet50.maxpool(x)
        x = self.resnet50.layer1(x)
        x = self.resnet50.layer2(x)
        x = self.resnet50.layer3(x)
        x = self.resnet50.layer4(x) # [64, 2048, 2, 8] : [B,C,H,W]
        x = self.resnet50.fc1(x) # [64, 256, 2, 8] : [B,C,H,W]
        x = x.permute(0, 3, 1, 2) # [64, 8, 256, 2] : [B,W,C,H]
       
        x = x.flatten(2) # [64, 8, 512] : [B,W,CH]
        x = x.permute(0,2,1) # [64,512,8]
        x = self.resnet50.fc2(x) # [64,512,64]
        x = x.permute(2, 0, 1) # [32, 1, 512] : [W,B,CH]
        # rnn features
        output = self.rnn(x)

        # add log_softmax to converge output
        output = F.log_softmax(output, dim=2)

        return output

    def backward_hook(self, module, grad_input, grad_output):
        for g in grad_input:
            g[g != g] = 0   # replace all nan/inf in gradients to zero
