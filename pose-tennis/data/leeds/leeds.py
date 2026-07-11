import os
import torch
import numpy as np
from scipy.io import loadmat
from skimage import transform
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
        joint_labels = np.transpose(np.array(joint_labels))

        sample = {'img': img, 'joint_labels': joint_labels}

        # Apply transforms
        if self.transform:
            sample = self.transform(sample)

        # Return a sample
        return sample
        

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        img, joint_labels = sample['img'], sample['joint_labels']

        h, w = img.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w, v = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(img, (new_h, new_w, img.shape[2]))

        # h and w are swapped for joint labels because for images,
        # x and y axes are axis 1 and 0 respectively
        joint_labels = joint_labels * [new_w / w, new_h / h, 1]

        return {'img': img, 'joint_labels': joint_labels}


class RandomCrop(object):
    """Randomly crop an image in a sample

    Args:
        output_size (tuple or int): Desired output size. If int, square crop is made.
    """
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 3
            self.output_size = output_size

    def __call__(self, sample):
        img, joint_labels = sample['img'], sample['joint_labels']

        h, w = img.shape[:2]
        new_h, new_w = self.output_size
        
        # If image is already smaller than the output size just return
        if h < new_h or w < new_w:
            return {'img': img, 'joint_labels': joint_labels}

        top = np.random.randint(0, h - new_h + 1)
        left = np.random.randint(0, w - new_w + 1)

        img = img[top: top + new_h,
                      left: left + new_w, :]

        joint_labels = joint_labels - [left, top, 0]

        return {'img': img, 'joint_labels': joint_labels}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        img, joint_labels = sample['img'], sample['joint_labels']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        img = img.transpose((2, 0, 1))
        return {'img': torch.from_numpy(img),
                'joint_labels': torch.from_numpy(joint_labels)}


class Normalize(object):
    """Normalize image using ImageNet stats (for ResNet transfer learning)."""
    def __init__(self, mean=None, std=None):
        self.mean = mean or [0.485, 0.456, 0.406]
        self.std = std or [0.229, 0.224, 0.225]
        self.normalize = transforms.Normalize(self.mean, self.std)

    def __call__(self, sample):
        img, joint_labels = sample['img'], sample['joint_labels']
        img = self.normalize(img)
        return {'img': img, 'joint_labels': joint_labels}


def show_joints(img, joint_labels):
    """Show image with joints on image"""
    # get position (x, y, visibility)
    x = joint_labels[:, 0]
    y = joint_labels[:, 1]
    v = joint_labels[:, 2]

    # Show image and place markers on joints
    plt.imshow(img)
    plt.scatter(x, y, 50, c="r", marker="+")

if __name__ == '__main__':
    leeds_dataset = LeedsSportsDataset('joints.mat', 'images')
    #print(len(leeds_dataset))
    #fig = plt.figure()

    #for i, sample in enumerate(leeds_dataset):
    #    print(f"{i}: img shape - {sample['img'].shape},\
    #            joint label shape: {sample['joint_labels'].shape}")
    #    ax = plt.subplot(1, 4, i+1)
    #    plt.tight_layout()
    #    ax.set_title('Sample #{}'.format(i))
    #    ax.axis('off')
    #    show_joints(**sample)
    #    if i == 0:
    #        plt.show()
    #        break

    #img = mpimg.imread('./leeds/images/im00001.jpg')
    #show_joints(img, joints[0])

    #plt.axis('off')
    #plt.show()

    scale = Rescale(256)
    crop = RandomCrop(128)
    composed = transforms.Compose([Rescale(256),
                               RandomCrop(224)])

    # Apply each of the above transforms on sample.
    fig = plt.figure()
    sample = leeds_dataset[0]
    for i, tsfrm in enumerate([scale, crop, composed]):
        transformed_sample = tsfrm(sample)
        ax = plt.subplot(1, 3, i + 1)
        plt.tight_layout()
        ax.set_title(type(tsfrm).__name__)
        show_joints(**transformed_sample)

    plt.show()
