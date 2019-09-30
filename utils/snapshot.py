import torch
from pathlib import Path
import os


def get_latest_sanpshot(dir, keyword=None):
    if isinstance(dir, str):
        dir = Path(dir)
    if keyword is None:
        files = list(dir.glob('*.pt'))
    else:
        files = list(dir.glob(f'*{keyword}*.pt'))
    if len(files) == 0:
        print('no snapshot found.')
        return None
    file_updates = {file_path: os.stat(
        file_path).st_mtime for file_path in files}
    latest_file_path = max(file_updates, key=file_updates.get)
    print(f'latest snapshot is {str(latest_file_path)}')
    return str(latest_file_path)


def save_snapshots(epoch, model, optimizer, scheduler, path):
    if isinstance(model, torch.nn.DataParallel):
        module = model.module
    else:
        module = model

    torch.save({
        'epoch': epoch + 1,
        'model': module.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': None if scheduler.__class__.__name__ == 'CyclicLR' else scheduler.state_dict()
    }, path)


def load_snapshots_to_model(path, model=None, optimizer=None, scheduler=None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(path, map_location=device)
    if model is not None:
        try:
            model.load_state_dict(checkpoint['model'])
        except: # when you transfer angular model to non-angular model
            checkpoint['model'].pop('fc_angular.weight', None)
            new_params = model.state_dict()
            new_params.update(checkpoint['model'])
            model.load_state_dict(new_params)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler'])


def load_epoch(path):
    checkpoint = torch.load(path)
    return checkpoint['epoch']


def transfer_snapshots_to_twoheadmodel(path, model=None, optimizer=None, scheduler=None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(path, map_location=device)
    if model is not None:
        checkpoint['model'].pop('fc_angular.weight', None)
        new_params = model.backbone.state_dict()
        new_params.update(checkpoint['model'])
        model.backbone.load_state_dict(new_params)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler'])
