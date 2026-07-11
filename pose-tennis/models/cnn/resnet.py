import os
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import matplotlib.pyplot as plt

import data.leeds.leeds as leeds

def imshow(img, joint_labels_batch, nrow):
    """Display image for Tensor."""
    img = img.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    img = np.clip(img, 0, 1)
    plt.imshow(img)
    x = y = np.array([])
    for i, joint_labels in enumerate(joint_labels_batch):
        joint_labels = joint_labels.numpy()
        # need to shift down x values
        sample_joints_x = joint_labels[:,0] + (i%nrow) * (224) # image width is reshaped to 224
        x = np.concatenate((x, sample_joints_x))
        # y values can stay the same
        sample_joints_y = joint_labels[:,1] + (i // nrow) * 224
        y = np.concatenate((y, sample_joints_y))
        
    print(x.shape, y.shape)
    plt.scatter(x, y, 50, c="r", marker="+")
    plt.show()

if __name__ == '__main__':
    leeds_transforms = transforms.Compose([
            leeds.Rescale(256),
            leeds.RandomCrop(224),
            leeds.ToTensor(),
            leeds.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    # get path (run from pose-tennis)
    current_dir = os.getcwd()
    data_path = os.path.join(current_dir, 'data', 'leeds')
    joints_path = os.path.join(data_path, 'joints.mat')
    images_path = os.path.join(data_path, 'images')

    leeds_dataset = leeds.LeedsSportsDataset(joints_path, images_path, transform=leeds_transforms)

    batch_size = 16
    validation_split = .2
    shuffle_dataset = True
    random_seed= 42

    dataset_size = len(leeds_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(leeds_dataset, batch_size=batch_size, 
                                            sampler=train_sampler)
    validation_loader = torch.utils.data.DataLoader(leeds_dataset, batch_size=batch_size,
                                                    sampler=valid_sampler)

    # Get a batch of training data
    sample_batch = next(iter(train_loader))
    img_batch = sample_batch['img']
    joint_labels_batch = sample_batch['joint_labels']
    # Make a grid from batch
    nrow = 8
    padding = 0
    out = torchvision.utils.make_grid(img_batch, nrow=nrow, padding=padding)

    imshow(out, joint_labels_batch, nrow)

    # Usage Example:
    # num_epochs = 10
    # for epoch in range(num_epochs):
    #     print(epoch)
    #     # Train:   
    #     for batch_index, sample in enumerate(train_loader):
    #         img, joint_labels = sample['img'], sample['joint_labels']
