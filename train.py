from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from paths import *
from config import *

def train(model, criterion, optimizer, train_loader, valid_loader, epochs):
    """
    params
    ---
    model : nn.Module
    optimizer : nn.Object
    criterion : nn.Object
    iterator : torch.utils.data.DataLoader
    epochs : int
    """
    for p in model.parameters():
        p.requires_grad = True
    model.train()

    for epoch in range(epochs):
      skipped = 0
      nan_number = 0
      total_cost = 0
      M = 0
      
      for item in tqdm(train_loader):
        X, Y = item
        X = X.cuda()
        Y = Y.cuda() 
        length = []
        for y in Y.transpose(0,1):
          len_i = 0
          for y_i in y:
            if y_i != 0:
              len_i += 1
            else:
              break
          length.append(len_i)
        length = torch.tensor(length)
        optimizer.zero_grad()
        preds = model(X)
        preds_size = Variable(torch.LongTensor([preds.size(0)] * batch_size))
        _, temp = preds.max(2)
        #text = ''
        #temp = temp.transpose(1, 0).contiguous().view(-1).data
        #for p in temp:
        #  text += str(alphabet[p])
      
        cost = criterion(preds, Y.transpose(0,1), preds_size, length) / batch_size
        if torch.isnan(cost) or torch.isinf(cost):
          del cost
        else:
          cost.backward()
          optimizer.step()
          total_cost += cost
          #print(cost)
          M += 1
          
      torch.save(model.state_dict(), PATH_TO_SAVE_MODEL)
      print('model has been saved')
      print(f'--epoch {epoch+1}--')
      print(f' train cost: {total_cost/M}')
      print('nan_number=',skipped)
      print('---')
      gc.collect()
