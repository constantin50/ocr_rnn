import os
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import sampler
import torchvision.transforms as transforms
from config import *

class OCRdataset(Dataset):
    def __init__(self, path_to_imgdir: str, path_to_labels: str, transform_list = None):
        super(OCRdataset, self).__init__()
        self.imgdir = path_to_imgdir
        df = pd.read_csv(path_to_labels, sep = '\t', names = ['image_name', 'label'])
        self.image2label = [(self.imgdir + image, label) for image, label in zip(df['image_name'], df['label'])]
        if transform_list == None:
          transform_list =  [transforms.Grayscale(1),
                              transforms.ToTensor(), 
                              transforms.Normalize((0.5,), (0.5,))]
        self.transform = transforms.Compose(transform_list)
        self.collate_fn = Collator()

    def __len__(self):
        return len(self.image2label)

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        image_path, label = self.image2label[index]
        img = Image.open(image_path)
        if self.transform is not None:
          img = self.transform(img)
        item = {'idx' : index, 'img': img, 'label': label}
        return item


class Collator(object):
    
    def __call__(self, batch):
        width = [item['img'].shape[2] for item in batch]
        indexes = [item['idx'] for item in batch]
        imgs = torch.ones([len(batch), batch[0]['img'].shape[0], batch[0]['img'].shape[1], 
                           max(width)], dtype=torch.float32)
        for idx, item in enumerate(batch):
            try:
                imgs[idx, :, :, 0:item['img'].shape[2]] = item['img']
            except:
                print(imgs.shape)
        item = {'img': imgs, 'idx':indexes}
        if 'label' in batch[0].keys():
            labels = [item['label'] for item in batch]
            item['label'] = labels
        return item