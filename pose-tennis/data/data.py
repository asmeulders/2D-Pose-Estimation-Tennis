import os
import torch
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image

plt.ion()

labels = np.load('./leeds/labels.npz')

print(labels['arr_0'])

data = loadmat('./leeds/joints.mat')

joints = data['joints']

print(joints.shape)

def show_joints(img, joints):
    plt.imshow(img)
    # plt.scatter(joints
    plt.pause(0.001)

img = mpimg.imread('./leeds/images/im00001.jpg')
plt.imshow(img)
plt.axis('off')
plt.show()
