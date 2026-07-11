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
        # Get joint labels from file
        data = loadmat(mat_file)
        joints = data['joints']
        joints = np.transpose(np.array(joints), (2,1,0))
        
        # Set values
        self.joints = joints
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.joints.shape[0] # 10000

    def __getitem__(self, idx):
        # Get images from leeds/images
        filename = f"im{idx+1:05d}.jpg"
        img_path = os.path.join(self.img_dir, filename)
        img = decode_image(img_path)
        img = np.transpose(np.array(img), (1,2,0)) # (chan, height, width) -> (height, width, chan)
        
        # Get joint labels
        joint_labels = self.joints[idx]

        # Apply transforms
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            joint_labels = self.target_transform(joint_labels)

        # Return a sample
        return {'img': img, 'joint_labels': joint_labels}
        

#data = loadmat('leeds/joints.mat')
#joints = data['joints']
#joints = np.transpose(np.array(joints), (2,1,0))

def show_joints(img, joint_labels):
    """Show image with joints on image"""
    # get position (x, y, visibility)
    x, y, v = joint_labels

    # Show image and place markers on joints
    plt.imshow(img)
    plt.scatter(x, y, 50, c="r", marker="+")

leeds_dataset = LeedsSportsDataset('leeds/joints.mat', 'leeds/images')
print(len(leeds_dataset))
fig = plt.figure()

for i, sample in enumerate(leeds_dataset):
    print(f"{i}: img shape - {sample['img'].shape},\
            joint label shape: {sample['joint_labels'].shape}")
    ax = plt.subplot(1, 4, i+1)
    plt.tight_layout()
    ax.set_title('Sample #{}'.format(i))
    ax.axis('off')
    show_joints(**sample)
    if i == 0:
        plt.show()
        break

#img = mpimg.imread('./leeds/images/im00001.jpg')
#show_joints(img, joints[0])

#plt.axis('off')
#plt.show()
