import os
import torch
from torch import nn
import torch.optim as optim
from torchvision import transforms

import data.leeds.leeds as leeds
from models.cnn.resnet import make_resnet50, train
from utils.train_utils import load_ckp

def train_fine_tune(model, split_sizes, data_loaders, resume=False):
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    loss_function = nn.MSELoss()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckp_dir = os.path.join('models', 'cnn', 'weights', 'checkpoints')
    best_path = os.path.join('models', 'cnn', 'weights', 'best_model')
    start_epoch = 0
    if resume:
        model, optimizer, start_epoch = load_ckp(os.path.join(ckp_dir, 'checkpoint.pth'), model, optimizer, device)
    else:
        model.to(device)
    
    model.to(device)
    print(device)
    train(model, 
          loss_function, optimizer, exp_lr_scheduler, 
          data_loaders, dataset_sizes=split_sizes, 
          checkpoint_dir=ckp_dir, best_model_dir=best_path,
          start_epoch=start_epoch, num_epochs=15, device=device)


def train_ffe(model, split_sizes, data_loaders, resume=False):
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                          lr=0.01, momentum=0.7)
    
    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    loss_function = nn.MSELoss()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckp_dir = os.path.join('models', 'cnn', 'weights', 'checkpoints')
    best_path = os.path.join('models', 'cnn', 'weights', 'best_model')
    start_epoch = 0
    if resume:
        model, optimizer, start_epoch = load_ckp(os.path.join(ckp_dir, 'checkpoint.pth'), model, optimizer, device)
    else:
        model.to(device)
    
    model.to(device)
    train(model, 
          loss_function, optimizer, exp_lr_scheduler, 
          data_loaders, dataset_sizes=split_sizes, 
          checkpoint_dir=ckp_dir, best_model_dir=best_path,
          start_epoch=start_epoch, num_epochs=30, device=device
        )

if __name__ == '__main__':
    leeds_transforms = transforms.Compose([
            leeds.Rescale(256),
            leeds.RandomCrop(224),
            leeds.ToTensor(),
            leeds.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    base_dir = os.getcwd()
    
    leeds_dataset = leeds.LeedsSportsDataset(
            mat_file=os.path.join(base_dir, 'data', 'leeds', 'joints.mat'), 
            img_dir=os.path.join(base_dir, 'data', 'leeds', 'images'), 
            transform=leeds_transforms)
    
    train_split, val_split, test_split = 0.75, 0.2, 0.05
    batch_size = 16
    random_seed = 42
    data_loaders = leeds.split_dataset(leeds_dataset, train_split, val_split, test_split,
                                       batch_size, random_seed, shuffle_dataset=True)
    
    model = make_resnet50(fine_tune=False, weights_path=os.path.join('models', 'cnn', 'weights', 'best_model', 'best_model.pth'))

    size = len(leeds_dataset)
    split_sizes = {
        'train': size * train_split,
        'val': size * val_split,
        'test': size * test_split
    }

    # train_fine_tune(model, split_sizes, data_loaders, resume=True)

    # model = make_resnet50(fine_tune=False)

    train_ffe(model, split_sizes, data_loaders)

    print('Done training')
