import torch
import tqdm
from utility import *
from config import *

def eval(model, test_loader)
  '''
  params
  ---
  model : nn.Module object
  
  test_loader : 
  
  returns
  ---
  accuracy : float
  '''
  correct = 0
  errors = []
  for item in tqdm(test_loader):
    X, Y = item
    X = X.cuda()
    Y = Y.cuda() 
    length = torch.tensor([len(Y.transpose(0,1)[0])]*1)
    preds = model(X)
    text_preds = decode(preds)
    text_true = ''.join([alphabet[p] for p in Y.transpose(1, 0)[0].data])
    if text_preds == text_true:
      correct += 1
    else:
      errors.append((text_preds, text_true))

  accuracy = correct/len(test_loader)
  return accuracy
