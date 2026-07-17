import os
import torch
import torchvision
from torchvision import transforms
import numpy as np

from utils.train_utils import load_ckp
import data.leeds.leeds as leeds
from models.cnn.resnet import make_resnet50, visualize_predictions

if __name__ == '__main__':
    leeds_transforms = transforms.Compose([
            leeds.Rescale(256),
            leeds.RandomCrop(224),
            leeds.ToTensor(),
            leeds.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    model = make_resnet50(fine_tune=False, weights_path=os.path.join('models', 'cnn', 'weights', 'best_model', 'best_model.pth'))
    visualize_predictions(model, 'Jimmy_Connors.jpg', leeds_transforms, device='cpu')