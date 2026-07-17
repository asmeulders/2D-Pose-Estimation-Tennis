import os, time, math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tempfile import TemporaryDirectory

import data.leeds.leeds as leeds
from utils.train_utils import save_ckp, load_ckp

def make_resnet50(fine_tune=True, weights_path=None, device='cpu'):
    # Get model
    initialize_weights = None if weights_path else ResNet50_Weights.DEFAULT
    model = resnet50(weights=initialize_weights)

    # Freeze parameters so gradients arent included
    if not fine_tune:
        for param in model.parameters():
            param.requires_grad = False
    
    # Make final output layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 28) # 14 joints * 2 coordinates

    # Load weights
    if weights_path:
        assert isinstance(weights_path, str)
        load_ckp(weights_path, model, device, optimizer=None)
    
    return model




def train(model, loss_function, optimizer, scheduler, dataloaders, dataset_sizes, checkpoint_dir, best_model_dir, start_epoch=0, num_epochs=25, device="cpu"):
    since = time.time()
    best_loss = float('inf')
    print(device)

    for epoch in range(start_epoch, num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0

            for sample_batch in dataloaders[phase]:
                img_batch = sample_batch['img'].to(device)
                joint_labels_batch = sample_batch['joint_labels'].to(device)  # shape: (batch, 14, 3) if visibility still included
                target = joint_labels_batch[:, :, :2]

                optimizer.zero_grad() # reset gradients

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(img_batch)
                    outputs = outputs.view(-1, 14, 2)

                    loss = loss_function(outputs, target)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * img_batch.size(0)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            print(f'{phase} Loss: {epoch_loss:.4f}')

            checkpoint = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }

            if phase == 'val':
                is_best = epoch_loss < best_loss
                if is_best:
                    best_loss = epoch_loss

                save_ckp(checkpoint, is_best, checkpoint_dir, best_model_dir)            


    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Loss: {best_loss:4f}')

    return model


def visualize_model(model, dataloaders, device="cpu"):
    was_training = model.training
    model.eval()

    subplot_size = math.ceil(math.sqrt(dataloaders['val'].batch_size))
    fig, axs = plt.subplots(subplot_size, subplot_size)
    fig.suptitle("Visualize model")

    with torch.no_grad():
        _, sample_batch = next(enumerate(dataloaders['val']))
        img_batch, joint_labels_batch = sample_batch['img'], sample_batch['joint_labels']
        img_batch, joint_labels_batch = img_batch.to(device), joint_labels_batch.to(device)

        outputs = model(img_batch)
        outputs = outputs.view(-1, 14, 2)
        outputs = outputs.numpy()

        for i, img in enumerate(img_batch):
            row, col = i // subplot_size, i % subplot_size
            show_img(img, outputs[i], axs[row, col])

        model.train(mode=was_training)

def visualize_predictions(model, img_path, transforms, device):
    was_training = model.training
    model.eval()

    img = np.asarray(Image.open(img_path))
    mock_sample = {
        'img': img,
        'joint_labels': np.zeros((14,3))
    }
    img = transforms(mock_sample)['img']
    img = img.to(device)

    with torch.no_grad():
        img = img.unsqueeze(0)
        outputs = model(img)

        joint_preds = outputs.numpy()
        joint_preds = joint_preds.reshape(14,2)

        ax = plt.subplot(2,2,1)
        ax.axis('off')
        show_img(img.cpu().data[0], joint_preds, ax=ax)

        plt.show()

        model.train(mode=was_training)


def show_img(img, joints, ax):
    img = img.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    img = np.clip(img, 0, 1)

    ax.imshow(img)
    ax.scatter(joints[:,0], joints[:,1], 50, c="r", marker="+")


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
    model_path = os.path.join(current_dir, 'models', 'cnn')
    
    leeds_dataset = leeds.LeedsSportsDataset(joints_path, images_path, transform=leeds_transforms)

    batch_size = 16
    validation_split = .2
    test_split = 0.05
    shuffle_dataset = True
    random_seed= 42

    dataset_size = len(leeds_dataset)
    indices = list(range(dataset_size))
    train_index = int(np.floor((validation_split + test_split) * dataset_size))
    validation_index = int(np.floor(test_split  * dataset_size))

    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    
    indices = {
        'train': indices[train_index:],
        'val': indices[validation_index:train_index],
        'test': indices[:validation_index]
    }

    # Creating PT data samplers and loaders:
    samplers = {
        'train': SubsetRandomSampler(indices['train']),
        'val': SubsetRandomSampler(indices['val']),
        'test': SubsetRandomSampler(indices['test'])
    }

    loaders = {
        'train': torch.utils.data.DataLoader(leeds_dataset, batch_size=batch_size, 
                                                    sampler=samplers['train']),
        'val': torch.utils.data.DataLoader(leeds_dataset, batch_size=batch_size,
                                                    sampler=samplers['val']),
        'test': torch.utils.data.DataLoader(leeds_dataset, batch_size=batch_size,
                                                    sampler=samplers['test'])
    }

    dataset_sizes = {
        'train': dataset_size * (1 - (validation_split + test_split)),
        'val': dataset_size * validation_split,
        'test': dataset_size * test_split
    }

    device = 'cpu'
    model = make_resnet50(fine_tune=True, weights_path=os.path.join(model_path, 'weights', 'best_model', 'best_model.pth'))
    model.to(device)

    visualize_model(model, loaders, device=device)
    plt.show()
