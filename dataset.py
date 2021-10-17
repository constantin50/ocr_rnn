import random
import torch
from torchvision import transforms
from config import *
from utility import *

# MAKE TEXT TO BE THE SAME LENGTH
class TextCollate():
    def __call__(self, batch):
        x_padded = []
        max_y_len = max([i[1].size(0) for i in batch])
        y_padded = torch.LongTensor(max_y_len, len(batch))
        y_padded.zero_()

        for i in range(len(batch)):
            x_padded.append(batch[i][0].unsqueeze(0))
            y = batch[i][1]
            y_padded[:y.size(0), i] = y

        x_padded = torch.cat(x_padded)
        return x_padded, y_padded

# store list of images' names (in directory) and does some operations with images
class TextLoader(torch.utils.data.Dataset):
    def __init__(self, images_name, labels, char2idx, idx2char, eval=False):
        """
        params
        ---
        images_name : list
            list of names of images (paths to images)
        labels : list
            list of labels to correspondent images from images_name list
        char2idx : dict
        idx2char : dict
        """
        self.images_name = images_name
        self.labels = labels
        self.char2idx = char2idx
        self.idx2char = idx2char
        self.eval = eval
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((imgH,imgW)),
            transforms.RandomAffine(10 ,None ,[0.6 ,1] ,3 ,fill=255),
            transforms.transforms.GaussianBlur(3, sigma=(0.1, 1.9)),
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        #while self.labels[index] in to_remove:
          #index += 1
        img = self.images_name[index]
        if not self.eval:
            img = self.transform(img)
            img = img / img.max()
            img = img ** (random.random() * 0.7 + 0.6)
        else:
            img = np.transpose(img, (2, 0, 1))
            img = img / img.max()
        
        label = text_to_labels(self.labels[index], self.char2idx)
        return (torch.FloatTensor(img), torch.LongTensor(label))

    def __len__(self):
        return len(self.labels)
