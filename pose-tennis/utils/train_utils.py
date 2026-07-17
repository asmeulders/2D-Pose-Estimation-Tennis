import os
import torch
import shutil

def save_ckp(state, is_best, checkpoint_dir, best_model_dir):
    print('Saving Model')
    f_path = os.path.join(checkpoint_dir, 'checkpoint.pth')
    torch.save(state, f_path)
    if is_best:
        best_fpath = os.path.join(best_model_dir, 'best_model.pth')
        shutil.copyfile(f_path, best_fpath)

def load_ckp(checkpoint_fpath, model, optimizer, device):
    print('Loading Model')
    checkpoint = torch.load(checkpoint_fpath, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer, checkpoint['epoch']
