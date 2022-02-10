import time
from tqdm import tqdm
import torch
import torchvision.transforms as transforms
from torch.nn.utils.clip_grad import clip_grad_norm_
from textdistance import levenshtein as lev

from config import *

# class for mapping symbols into indicies and vice versa
class LabelCoder(object):
    def __init__(self, alphabet, ignore_case=False):
        self.alphabet = alphabet
        self.char2idx = {}
        for i, char in enumerate(alphabet):
            self.char2idx[char] = i + 1
        self.char2idx[''] = 0

    def encode(self, text: str):
        length = []
        result = []
        for item in text:
            length.append(len(item))
            for char in item:
                if char in self.char2idx:
                    index = self.char2idx[char]
                else:
                    index = 0
                result.append(index)

        text = result
        return (torch.IntTensor(text), torch.IntTensor(length))

    def decode(self, t, length, raw=False):
        if length.numel() == 1:
            length = length[0]
            assert t.numel() == length, "text with length: {} does not match declared length: {}".format(t.numel(),
                                                                                                         length)
            if raw:
                return ''.join([self.alphabet[i - 1] for i in t])
            else:
                char_list = []
                for i in range(length):
                    if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                        char_list.append(self.alphabet[t[i] - 1])
                return ''.join(char_list)
        else:
            # batch mode
            assert t.numel() == length.sum(), "texts with length: {} does not match declared length: {}".format(
                t.numel(), length.sum())
            texts = []
            index = 0
            for i in range(length.numel()):
                l = length[i]
                texts.append(
                    self.decode(t[index:index + l], torch.IntTensor([l]), raw=raw))
                index += l
            return texts


def predict(model, imagedir: str):
  transform_list =  [transforms.Grayscale(1),
                    transforms.ToTensor(), 
                    transforms.Normalize((0.5,), (0.5,))]
  transform = transforms.Compose(transform_list)
  result = {'image_name' : [], 'pred_label' : []}
  for filename in tqdm(os.listdir(imagedir)):
    img = Image.open(imagedir + filename)
    img = transform(img).unsqueeze(0)
    logits = model(img.to(DEVICE))
    logits = logits.contiguous().cpu()
    T, B, H = logits.size()
    pred_sizes = torch.LongTensor([T for i in range(B)])
    probs, preds = logits.max(2)
    preds = preds.transpose(1, 0).contiguous().view(-1)
    sim_preds = coder.decode(preds.data, pred_sizes.data, raw=False)
    result['image_name'].append(filename)
    result['pred_label'].append(sim_preds)
  return result


def fit(model, optimizer, loss_fn, loader, epochs = 12):
    coder = LabelCoder(ALPHABET)
    print('epoch | mean loss | mean cer | mean wer ')
    for epoch in range(epochs):
        model.train()
        outputs = []
        for batch_nb, batch in enumerate(loader):
            
            input_, targets = batch['img'], batch['label']
            targets, lengths = coder.encode(targets)
            logits = model(input_.to(DEVICE))
            logits = logits.contiguous().cpu()
            T, B, H = logits.size()
            pred_sizes = torch.LongTensor([T for i in range(B)])
            targets = targets.view(-1).contiguous()
            loss = loss_fn(logits, targets, pred_sizes, lengths)
            probs, preds = logits.max(2)
            preds = preds.transpose(1, 0).contiguous().view(-1)
            sim_preds = coder.decode(preds.data, pred_sizes.data, raw=False)

            char_error = sum([lev(batch['label'][i], sim_preds[i])/max(len(batch['label'][i]), len(sim_preds[i])) for i in range(len(batch['label']))])/len(batch['label'])
            word_error = 1 - sum([batch['label'][i] == sim_preds[i] for i in range(len(batch['label']))])/len(batch['label'])

            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), 0.05)
            optimizer.step()
            output = {'loss': abs(loss.item()),'cer': char_error,'wer': word_error}
        
            outputs.append(output)
        mean_loss = sum([outputs[i]['loss'] for i in range(len(outputs))])/len(outputs)
        char_error = sum([outputs[i]['cer'] for i in range(len(outputs))])/len(outputs)
        word_error = sum([outputs[i]['wer'] for i in range(len(outputs))])/len(outputs)
        print(epoch, ' '*5 ,"%.3f" % mean_loss,' '*5, "%.3f" % char_error,' '*5, "%.3f" % word_error)

def evaluate(model):
  coder = LabelCoder(ALPHABET)
  labels, predictions = [], []
  for iteration, batch in enumerate(tqdm(loader)):
      input_, targets = batch['img'].to(DEVICE), batch['label']
      labels.extend(targets)
      targets, _ = coder.encode(targets)
      logits = model(input_)
      logits = logits.contiguous().cpu()
      T, B, H = logits.size()
      pred_sizes = torch.LongTensor([T for i in range(B)])
      probs, pos = logits.max(2)
      pos = pos.transpose(1, 0).contiguous().view(-1)
      sim_preds = coder.decode(pos.data, pred_sizes.data, raw=False)
      predictions.extend(sim_preds)
  char_error = sum([lev(labels[i], predictions[i]) for i in range(len(labels))])/len(labels)
  word_error = 1 - sum([labels[i] == predictions[i] for i in range(len(labels))])/len(labels)
  return {'char_error' : char_error, 'word_error' : word_error}