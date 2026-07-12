import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import matplotlib.pyplot as plt
import time
from tempfile import TemporaryDirectory

import data.leeds.leeds as leeds


def train(model, loss_function, optimizer, scheduler, dataloaders, dataset_sizes, num_epochs=25, device="cpu"):
    since = time.time()
    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')

        torch.save(model.state_dict(), best_model_params_path)
        best_loss = float('inf')

        for epoch in range(num_epochs):
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

                if phase == 'val' and epoch_loss < best_loss:
                    print('update loss')
                    best_loss = epoch_loss
                    torch.save(model.state_dict(), best_model_params_path)

            print()

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Loss: {best_loss:4f}')

        model.load_state_dict(torch.load(best_model_params_path, weights_only=True))
    return model


def visualize_model(model, dataloaders, num_images=6, device="cpu"):
    was_training = model.training
    model.eval()
    # images_so_far = 0
    nrow = 8
    padding = 0
    # fig = plt.figure()

    with torch.no_grad():
        for i, sample_batch in enumerate(dataloaders['val']):
            img_batch = sample_batch['img']
            joint_labels_batch = sample_batch['joint_labels']
            img_batch = img_batch.to(device)
            joint_labels_batch = joint_labels_batch.to(device)

            # make inference
            outputs = model(img_batch)

            # resize from (1,24) to (2,14) (should be (x,y) in each row)
            outputs = outputs.numpy()
            outputs = np.resize(outputs, (2,14))
            print(outputs)

            out = torchvision.utils.make_grid(img_batch, nrow=nrow, padding=padding)

            imshow(out, outputs, nrow)

            # for j in range(inputs.size()[0]):
                
            #     images_so_far += 1
            #     ax = plt.subplot(num_images//2, 2, images_so_far)
            #     ax.axis('off')
            #     imshow(img_batch)

            #     if images_so_far == num_images:
            #         model.train(mode=was_training)
            #         return
        model.train(mode=was_training)

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

    # Get a batch of training data
    sample_batch = next(iter(loaders['train']))
    img_batch = sample_batch['img']
    joint_labels_batch = sample_batch['joint_labels']
    # Make a grid from batch
    nrow = 8
    padding = 0
    out = torchvision.utils.make_grid(img_batch, nrow=nrow, padding=padding)

    imshow(out, joint_labels_batch, nrow)


    # Download pretrained model and change output to my class (x,y) * 14 joints
    model = torchvision.models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 28)

    loss_function = nn.MSELoss()

    # Observe that all parameters are being optimized
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model = train(model, loss_function, optimizer, scheduler=exp_lr_scheduler,
                  dataloaders=loaders, dataset_sizes=dataset_sizes, num_epochs=1, device=device)
    
    torch.save(model.state_dict(), os.path.join(model_path, 'resnet_weights.pth'))

    visualize_model(model, loaders, device=device)
    plt.show()

    # Usage Example:
    # num_epochs = 10
    # for epoch in range(num_epochs):
    #     print(epoch)
    #     # Train:   
    #     for batch_index, sample in enumerate(train_loader):
    #         img, joint_labels = sample['img'], sample['joint_labels']
