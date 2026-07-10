import os
import torch
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision.io import decode_image
from PIL import Image


class LeedsSportsDataset(Dataset):
    """Leeds Sports Pose (Extended) Dataset"""

    def __init__(self, mat_file, img_dir, transform=None, target_transform=None):
        """
        Arguments:
            mat_file (string): Path to mat file with joint annotations.
            root_dir (string): Directory with images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        data = loadmat(mat_file)
        joints = data['joints']
        joints = np.transpose(np.array(joints), (2,1,0))

        self.joints = joints
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.joints.shape[0]

    def __getitem__(self, idx):
        filename = f"im{idx:05d}.jpg"
        img_path = os.path.join(self.img_dir, filename)
        image = decode_image(img_path)
        joint_labels = self.joints[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            joint_labels = self.target_transform(joint_labels)
        return {'image': image, 'joint_labels': joint_labels}
        

#data = loadmat('leeds/joints.mat')
#joints = data['joints']
#joints = np.transpose(np.array(joints), (2,1,0))

def show_joints(img, joints):
    """Show image with joints on image"""
    x, y, v = joints

    plt.imshow(img)
    plt.scatter(x, y, 50, c="r", marker="+")

leeds_dataset = LeedsSportsDataset('leeds/joints.mat', 'leeds/images')

fig = plt.figure()

for i, sample in enumerate(leeds_dataset):
    print(f"{i}: img shape - {sample['image'].shape},\
            joint label shape: {sample['joint_labels'].shape}")
    ax = plt.subplot(1, 4, i+1)
    plt.tight_layout()
    ax.set_title('Sample #{}'.format(i))
    ax.axis('off')
    show_landmarks(**sample)
    if i == 3:
        plt.show()
        break

#img = mpimg.imread('./leeds/images/im00001.jpg')
#show_joints(img, joints[0])

#plt.axis('off')
#plt.show()
