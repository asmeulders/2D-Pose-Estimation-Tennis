import os
import torch
import torch.nn as nn

import data.leeds.leeds as leeds


if __name__ == '__main__':
    current_dir = os.getcwd()
    data_path = os.path.join(current_dir, 'data', 'leeds')
    joints_path = os.path.join(data_path, 'joints.mat')
    images_path = os.path.join(data_path, 'images')
    leeds_dataset = leeds.LeedsSportsDataset(joints_path, images_path)

    print(len(leeds_dataset))
