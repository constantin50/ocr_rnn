from tqdm import tqdm
import dataset
from config import *
from dataset import *
import cv2
import numpy as np
import random
import os
import gc

# convert images and labels into defined data structures
def process_data(image_dir, labels_dir, ignore=[]):
    """
    params
    ---
    image_dir : str
      path to directory with images
    labels_dir : str
      path to tsv file with labels
    returns
    ---
    img2label : dict
      keys are names of images and values are correspondent labels
    chars : list
      all unique chars used in data
    all_labels : list
    """

    chars = []
    img2label = dict()

    raw = open(labels_dir, 'r', encoding='utf-8').read()
    temp = raw.split('\n')
    for t in temp:
        try:
            x = t.split('\t')
            flag = False
            for item in ignore:
                if item in x[1]:
                    flag = True
            if flag == False:
                img2label[image_dir + x[0]] = x[1]
                for char in x[1]:
                    if char not in chars:
                        chars.append(char)
        except:
            print('ValueError:', x)
            pass

    all_labels = sorted(list(set(list(img2label.values()))))
    chars.sort()
    chars = ['SOS'] + chars

    return img2label, chars, all_labels

    
def text_to_labels(s, char2idx):
    return [char2idx[i] for i in s if i in char2idx.keys()]
  
# SPLIT DATASET INTO TRAIN AND VALID PARTS
def train_valid_split(img2label, val_part=0.3):
    """
    params
    ---
    img2label : dict
        keys are paths to images, values are labels (transcripts of crops)
    returns
    ---
    imgs_val : list of str
        paths
    labels_val : list of str
        labels
    imgs_train : list of str
        paths
    labels_train : list of str
        labels
    """

    imgs_val, labels_val = [], []
    imgs_train, labels_train = [], []

    N = int(len(img2label) * val_part)
    items = list(img2label.items())
    random.shuffle(items)
    for i, item in enumerate(items):
        if i < N:
            imgs_val.append(item[0])
            labels_val.append(item[1])
        else:
            imgs_train.append(item[0])
            labels_train.append(item[1])
    print('valid part:{}'.format(len(imgs_val)))
    print('train part:{}'.format(len(imgs_train)))
    return imgs_val, labels_val, imgs_train, labels_train

# GENERATE IMAGES FROM FOLDER
def generate_data(img_paths):
    """
    params
    ---
    names : list of str
        paths to images
    returns
    ---
    data_images : list of np.array
        images in np.array format
    """
    data_images = []
    for path in tqdm(img_paths):
        img = cv2.imread(path)
        try:
            img = process_image(img)
            data_images.append(img.astype('uint8'))
        except:
            print(path)
            img = process_image(img)
    return data_images

# RESIZE AND NORMALIZE IMAGE
def process_image(img):
    """
    params:
    ---
    img : np.array
    returns
    ---
    img : np.array
    """
    w, h, _ = img.shape
    new_w = imgH
    new_h = int(h * (new_w / w))
    img = cv2.resize(img, (new_h, new_w))
    w, h, _ = img.shape

    img = img.astype('float32')

    new_h = imgW
    if h < new_h:
        add_zeros = np.full((w, new_h - h, 3), 255)
        img = np.concatenate((img, add_zeros), axis=1)

    if h > new_h:
        img = cv2.resize(img, (new_h, new_w))

    return img

def get_loaders(path_to_images, path_to_transcript, alphabet=alphabet, batch_size=batch_size):
  """
  params
  ---
  path_to_images : string
    paths to the folder with png/jpg images

  path_to_transcript : string
    path to tsv file with transcript in the following format:
    image1.jpg  transcript1
    image2.jpg  transcript2
    ...         ...
  
  batch_size : int

  returns
  ---

  """

  img2label, chars, all_words = process_data(path_to_images, path_to_transcript)
  X, y = [], []
  char2idx = {char: idx for idx, char in enumerate(alphabet)}
  idx2char = {idx: char for idx, char in enumerate(alphabet)}
  items = list(img2label.items())
  random.shuffle(items)
  for i, item in enumerate(items):
      X.append(item[0])
      y.append(item[1])
  X = generate_data(X)
  train_dataset = dataset.TextLoader(X, y, char2idx, idx2char, eval=False)
  train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True,
                                           batch_size=batch_size, pin_memory=True,
                                           drop_last=True, collate_fn=dataset.TextCollate())
  del train_dataset, X, y
  gc.collect()
  return train_loader

# decode predictions of a model; get characters
def decode(preds):
  text = ''
  _, preds = preds.max(2)
  preds = preds.transpose(1, 0).contiguous().view(-1).data
  previous = None
  for p in preds:
    #print(p)
    if p != previous and p != 0:
      text += str(alphabet[p])
      previous = p
  return text
