import pandas as pd
import numpy as np
import torch
from torch import nn
import torchvision.transforms as transforms
import cv2

class Vignetting(object):
    def __init__(self,
                 p = 0.1,
                 ratio_min_dist=0.2,
                 range_vignette=(0.2, 0.8),
                 random_sign=False):
        self.ratio_min_dist = ratio_min_dist
        self.range_vignette = np.array(range_vignette)
        self.random_sign = random_sign
        self.p = p

    def __call__(self, X, Y=None):

        if np.random.binomial(1, self.p) == 0:
          return X 
        h, w = X.shape[1:]
        min_dist = np.array([h, w]) / 2 * np.random.random() * self.ratio_min_dist
        # create matrix of distance from the center on the two axis
        x, y = np.meshgrid(np.linspace(-w / 2, w / 2, w), np.linspace(-h / 2, h / 2, h))
        x, y = np.abs(x), np.abs(y)
        # create the vignette mask on the two axis
        x = (x - min_dist[0]) / (np.max(x) - min_dist[0])
        x = np.clip(x, 0, 1)
        y = (y - min_dist[1]) / (np.max(y) - min_dist[1])
        y = np.clip(y, 0, 1)
        # then get a random intensity of the vignette
        vignette = (x + y) / 2 * np.random.uniform(*self.range_vignette)
        vignette = np.tile(vignette[None, ...], [1, 1, 1])

        sign = 2 * (np.random.random() < 0.5) * (self.random_sign) - 1
        Z = X * (1 + sign * vignette)
        return Z


class LensDistortion(object):
    def __init__(self, p = 0.1 ,d_coef=(0.15, 0.05, 0.05, 0.05, 0.05)):
        self.d_coef = np.array(d_coef)
        self.p = p

    def __call__(self, X):
        if np.random.binomial(1, self.p) == 0:
          return X 

        # get the height and the width of the image
        h, w = X.shape[:2]

        # compute its diagonal
        f = (h ** 2 + w ** 2) ** 0.5

        # set the image projective to carrtesian dimension
        K = np.array([[f, 0, w / 2],
                      [0, f, h / 2],
                      [0, 0, 1]])

        d_coef = self.d_coef * np.random.random(5)  # value
        d_coef = d_coef * (2 * (np.random.random(5) < 0.5) - 1)  # sign
        # Generate new camera matrix from parameters
        M, _ = cv2.getOptimalNewCameraMatrix(K, d_coef, (w, h), 0)

        # Generate look-up tables for remapping the camera image
        remap = cv2.initUndistortRectifyMap(K, d_coef, None, M, (w, h), 5)

        # Remap the original image to a new image
        Z = cv2.remap(np.float32(X.numpy()), *remap, cv2.INTER_LINEAR)
        return torch.from_numpy(Z)
