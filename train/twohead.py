'''
Copy this script to project home directory to use
'''

from utils.logger import *
from utils.dataset import *
from utils.utils import *
from utils.snapshot import *
from utils.image_utils import *
from utils.optimizers import *
from models.backbones import *
from models.metrics import *
from models.angular import *

import os
import sys
import time
import datetime
import argparse
import re
from pathlib import Path
from tqdm import tqdm

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import torchvision.models as models
from torchvision import transforms as T
from torch.autograd import Variable
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from sklearn.model_selection import train_test_split, GroupShuffleSplit, GroupKFold


EMBEDDING = {
    'alpha': 256,
    'baseline': 256,
    'chi': 1000,
    'sigma': 512,
}

DROPOUT = {
    'alpha': 0,
    'baseline': 0,
    'chi': 0.2,
    'sigma': 0,
}


class TwoHeadModel(nn.Module):
    '''
    Two head model

    Experiment image   --->( Backbone )---+
                                          |
                                          (-)--->( fc | angular )
                                          |
    Neg control image  --->( Backbone )---+
    '''

    def __init__(self, base_model, in_channel=6, embedding_size=256,
                 num_classes=1108, angular=True, dropout=0):
        super(TwoHeadModel, self).__init__()

        self.backbone = base_model
        self.embedding_size = embedding_size
        self.dropout = nn.Dropout(dropout)

        if angular:
            self.fc = AngleSimpleLinear(
                self.embedding_size, num_classes)
        else:
            self.fc = nn.Linear(
                self.embedding_size, num_classes)

    def forward(self, x_exp, x_con):

        x = self.backbone(x_exp) - self.backbone(x_con)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        y = self.fc(x)

        return y

    def set_dropout_ratio(self, ratio):
        assert 0 <= ratio < 1.

    def lock_backbones(self):
        for name, child in self.backbone.named_children():
            for param in child.parameters():
                param.requires_grad = False

    def unlock_backbones(self):
        for name, child in self.backbone.named_children():
            for param in child.parameters():
                param.requires_grad = True


def parse_result(data):
    res = []
    for batch in data:
        res.extend(batch)
    return res


def mixup_criterion(criterion, pred, y, cy, c):
    c = c.float()
    return c * criterion(pred, y) + (1 - c) * criterion(pred, cy)


def predict_test(export_path, model):
    site1_full = np.zeros((len(sub), NUM_CLASSES), dtype=np.float16)
    site2_full = np.zeros((len(sub), NUM_CLASSES), dtype=np.float16)

    print(f'Prediction in progress...Saving to {export_path}')

    for pred_epoch in range(TEST_AUG_EPOCH):
        site1 = []
        site2 = []

        # model.eval()
        with torch.no_grad():
            # for batch_i, (exp_imgs1, con_imgs1, exp_imgs2, con_imgs2, idx) in enumerate(loader_test):
            for batch_i, x_all in enumerate(loader_test):
                for site in [1, 2]:
                    exp_imgs = x_all[site*2-2].to(device)
                    con_imgs = x_all[site*2-1].to(device)

                    output = model(exp_imgs, con_imgs)
                    output = output.data.cpu().numpy()

                    if site == 1:
                        site1.append(output)
                    elif site == 2:
                        site2.append(output)

        site1 = np.array(parse_result(site1), dtype=np.float16)
        site2 = np.array(parse_result(site2), dtype=np.float16)

        site1_full[idxs, :] += site1/TEST_AUG_EPOCH
        site2_full[idxs, :] += site2/TEST_AUG_EPOCH

    np.save(export_path, (site1_full + site2_full) / 2)


def trainer(fold_i):
    # Model
    if opt.no_metric:
        base_model = get_model(opt.model, EMBEDDING[opt.model], angular=False, feature=True)
        model = TwoHeadModel(base_model, in_channel=6, embedding_size=EMBEDDING[opt.model],
                             num_classes=NUM_CLASSES, angular=False, dropout=DROPOUT[opt.model])
    else:
        base_model = get_model(opt.model, EMBEDDING[opt.model], angular=True, feature=True)
        model = TwoHeadModel(base_model, in_channel=6, embedding_size=EMBEDDING[opt.model],
                             num_classes=NUM_CLASSES, angular=True, dropout=DROPOUT[opt.model])
    model.to(device)

    step_size = 4
    base_lr, max_lr = DEFAULT_LR, 5 * DEFAULT_LR
    optimizer = RAdam(filter(lambda p: p.requires_grad, model.parameters()), lr=DEFAULT_LR)
    if opt.clr:
        scheduler = CyclicLR(optimizer, base_lr=base_lr, max_lr=max_lr,
                             step_size=step_size, mode='exp_range', gamma=0.96)
    else:
        scheduler = StepLR(optimizer, step_size=DEFAULT_LR_STEP, gamma=0.1)

    if opt.no_metric:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = AMSoftmaxLoss(margin_type=opt.margin, mixup=opt.mixup)
    early_stopping = EarlyStopping(patience=EARLY_STOPPING_ROUNDS, verbose=True)

    start_epoch = 0

    if opt.from_file != '':
        path = opt.from_file
        try:
            load_snapshots_to_model(path, model)
        except:
            transfer_snapshots_to_twoheadmodel(path, model)
        start_epoch = load_epoch(path)
        print(f'{path} loaded.')
    # XOR
    if opt.resume:
        path = get_latest_sanpshot(result_path, f'{opt.experiment}_{fold_i}')
        if path is not None:
            try:
                load_snapshots_to_model(path, model)
            except:
                transfer_snapshots_to_twoheadmodel(path, model)
            start_epoch = load_epoch(path)
            print(f'{path} loaded.')

    if torch.cuda.device_count() > 1:
        print('{} gpus found.'.format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)

    for epoch in range(start_epoch, start_epoch + opt.epoch):
        start_time = time.time()
        if opt.clr:
            if scheduler: # CyclicLR
                scheduler.batch_step()
        else:
            scheduler.step()

        if epoch - start_epoch == 0:
            if isinstance(model, torch.nn.DataParallel):
                model.module.lock_backbones()
            else:
                model.lock_backbones()
            print('Backbone locked.')
        elif epoch - start_epoch == 3:
            if isinstance(model, torch.nn.DataParallel):
                model.module.unlock_backbones()
            else:
                model.unlock_backbones()
            print('Backbone unlocked.')

        model.train()
        train_losses = []
        train_accuracies = []
        for i in range(2):
            for batch_i, (exp_imgs, con_imgs, labels) in enumerate(loader[i]):
                batches_done = len(loader[i]) * epoch * 2 + i * len(loader[i]) + batch_i

                exp_imgs = exp_imgs.to(device)
                con_imgs = con_imgs.to(device)

                output = model(exp_imgs, con_imgs)

                if opt.mixup:
                    for l in range(len(labels)):
                        labels[l] = labels[l].to(device)
                    loss = mixup_criterion(criterion, output, *labels).mean()
                else:
                    labels = labels.to(device)
                    loss = criterion(output, labels)

                loss.backward()

                if batches_done % opt.gradient_accumulations == 0:
                    # Accumulates gradient before each step
                    optimizer.step()
                    optimizer.zero_grad()


                if batch_i % LOG_INTERVAL == 0:
                    # Log progress
                    log_str = "\n---- [Epoch %d/%d, Batch %d/%d] ----\n" % (
                        epoch, start_epoch + opt.epoch, batch_i, len(loader[i]))

                    output = output.data.cpu().numpy()
                    output = np.argmax(output, axis=1) # get label
                    if opt.mixup:
                        acc = 0.0
                    else:
                        labels = labels.data.cpu().numpy()
                        acc = np.mean((output == labels).astype(int))
                    for param_group in optimizer.param_groups:
                        learing_rate = param_group['lr']
                    # Tensorboard logging
                    tensorboard_log = []
                    tensorboard_log += [(f"accuracy_{fold_i}", acc)]
                    tensorboard_log += [(f"loss_{fold_i}", loss.item())]
                    tensorboard_log += [(f"lr_{fold_i}", learing_rate)]
                    logger.list_of_scalars_summary(tensorboard_log, batches_done)

                    # Determine approximate time left for epoch
                    epoch_batches_left = len(loader[i]) - (batch_i + 1)
                    time_left = datetime.timedelta(
                        seconds=epoch_batches_left * (time.time() - start_time) / (batch_i + 1))
                    log_str += f"\n---- ETA {time_left}"

                    # Average evaluation
                    train_accuracies.append(acc)
                    train_losses.append(loss.item())

                    # print(log_str)

        log_str = "\n---- [Training Epoch %d/%d] ----\n" % (epoch, start_epoch + opt.epoch)
        average_loss_train = np.average(train_losses)
        average_acc_train = np.average(train_accuracies)
        log_str += f"loss {average_loss_train:.6f}\t accuracy {average_acc_train:.6f}"
        print(log_str)


        '''
        Prediction
        '''
        if opt.full_train:
            if epoch % SAVE_INTERVAL == 0:
                predict_test(result_path/'full'/f'{opt.experiment}_{epoch}', model)
                save_snapshots(epoch, model, optimizer, scheduler,
                               result_path/'full'/f'{opt.experiment}_{epoch}.pt')
            continue # no validation


        '''
        Validation
        '''
        if epoch % opt.evaluation_interval == 0:
            valid_losses = []
            valid_accuracies = []
            # model.eval()
            with torch.no_grad():
                for val_epoch in range(VALID_AUG_EPOCH):
                    for i in range(2):
                        for exp_imgs, con_imgs, labels in loader_val[i]:
                            exp_imgs = exp_imgs.to(device)
                            con_imgs = con_imgs.to(device)

                            output = model(exp_imgs, con_imgs)

                            labels = labels.to(device)
                            loss = criterion(output, labels)

                            output = output.data.cpu().numpy()
                            output = np.argmax(output, axis=1)
                            labels = labels.data.cpu().numpy()
                            acc = np.mean((output == labels).astype(int))
                            valid_accuracies.append(acc)
                            valid_losses.append(loss.item())

            # Evaluate the model on the validation set
            log_str = "\n---- [Validation Epoch %d/%d] ----\n" % (epoch, start_epoch + opt.epoch)
            average_loss_valid = np.average(valid_losses)
            average_acc_valid = np.average(valid_accuracies)
            log_str += f"loss {average_loss_valid:.6f}\t accuracy {average_acc_valid:.6f}\t time {(time.time() - start_time):.6f}"
            evaluation_metrics = [
                (f"val_accuracy_{fold_i}", average_acc_valid),
                (f"valid_loss_{fold_i}", average_loss_valid)
            ]
            logger.list_of_scalars_summary(evaluation_metrics, epoch)
            print(log_str)

            # EarlyStopping!
            if opt.stopping_target == 'loss':
                early_stopping_target = average_loss_valid
            elif opt.stopping_target == 'score':
                early_stopping_target = average_acc_valid
            else:
                early_stopping_target = average_loss_valid

            if early_stopping(early_stopping_target, model): # score updated
                save_snapshots(epoch, model, optimizer, scheduler,
                               result_path/f'best_{opt.experiment}_{fold_i}.pt')

        if early_stopping.early_stop:
            print("Early stopping!")
            load_snapshots_to_model(str(result_path/f'best_{opt.experiment}_{fold_i}.pt'), model.module)
            predict_test(result_path/f'best_{opt.experiment}_{fold_i}', model)
            break
    else:
        load_snapshots_to_model(str(result_path/f'best_{opt.experiment}_{fold_i}.pt'), model.module)
        predict_test(result_path/f'best_{opt.experiment}_{fold_i}', model)


if __name__ == "__main__":
    #%%
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="root path to data")
    parser.add_argument("--model", type=str, default="baseline",
                        help="model type")
    parser.add_argument("--no_metric", action='store_true',
                        help="use normal fc layer")
    parser.add_argument("--img_size", type=int, default=384, help="image size")
    parser.add_argument("--epoch", type=int, default=100,
                        help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="size of each image batch")
    parser.add_argument("--default_lr", type=float, default=5e-4,
                        help="default learning rate")
    parser.add_argument("--gradient_accumulations", type=int, default=2,
                        help="number of gradient accums before step")
    parser.add_argument("--resume", action='store_true',
                        help="continue from latest checkpoint model")
    parser.add_argument("--from_file", type=str, default='',
                        help="continue from specified checkpoint model (must be combined with --resume)")
    parser.add_argument("--n_cpu", type=int, default=8,
                        help="number of cpu threads to use during batch generation")
    parser.add_argument("--evaluation_interval", type=int, default=1,
                        help="interval evaluations on validation set")
    parser.add_argument("--experiment", type=str, default='all',
                        help="train submodels")
    parser.add_argument("--margin", type=str, default='cos',
                        help="cos or arc")
    parser.add_argument("--stopping_target", type=str, default='loss',
                        help="loss or score")
    parser.add_argument("--stopping_round", type=int, default=5,
                        help="early stopping round")
    parser.add_argument("--augmentation", type=int, default=0,
                        help="train with data augmentation")
    parser.add_argument("--mixup", action='store_true',
                        help="do mixup augmentation")
    parser.add_argument("--clr", action='store_true',
                        help="use cyclic LR")
    parser.add_argument("--full_train", action='store_true',
                        help="no validation set :)")
    parser.add_argument("--cv", type=int, default=3,
                        help="cross validation fold")
    parser.add_argument("--parallel", type=int, default=None,
                        help="parallel cross validation")
    opt = parser.parse_args()
    print(opt)


    #%%
    print(f'Training {opt.model}{opt.img_size}')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    SEED = 2019
    NUM_CLASSES = 1108
    DEFAULT_LR = opt.default_lr
    DEFAULT_LR_STEP = 10
    DEFAULT_LR_DECAY = 0.95
    DEFAULT_WEIGHT_DECAY = 5e-4
    EARLY_STOPPING_ROUNDS = opt.stopping_round
    LOG_INTERVAL = 100
    SAVE_INTERVAL = 5
    if opt.augmentation:
        VALID_AUG_EPOCH = 4
        TEST_AUG_EPOCH = 8
    else:
        VALID_AUG_EPOCH = 1
        TEST_AUG_EPOCH = 1
    torch.manual_seed(SEED)

    data_path = Path(opt.data)
    result_path = Path(f'results/twohead_{opt.model}{opt.img_size}{"_fc" if opt.no_metric else ""}')
    result_path.mkdir(exist_ok=True)
    if opt.full_train:
        (result_path/'full').mkdir(exist_ok=True)
    log_path = Path(f'logs/twohead_{opt.model}_{opt.img_size}_{opt.experiment}{"_full" if opt.full_train else ""}{"_nometric" if opt.no_metric else ""}')
    log_path.mkdir(exist_ok=True)
    logger = Logger(str(log_path))
    sub = pd.read_csv(data_path/'sample_submission.csv')


    '''
    Train!
    '''
    if opt.cv in [0, 1]:

        # Get dataloader
        df = pd.read_csv(data_path / 'train.csv')
        df_test = pd.read_csv(data_path / 'test.csv')
        control = pd.read_csv(data_path/'train_controls.csv')
        control_test = pd.read_csv(data_path/'test_controls.csv')

        if opt.experiment != 'all': # filter exp
            df = df.loc[df.experiment.str.contains(opt.experiment)]
            df_test = df_test.loc[df_test.experiment.str.contains(opt.experiment)]
            control = control.loc[control.experiment.str.contains(opt.experiment)]
            control_test = control_test.loc[control_test.experiment.str.contains(opt.experiment)]

        if not opt.full_train:
            gss = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=SEED)
            gss.get_n_splits()
            train_idx, valid_idx = list(gss.split(df, groups=df.experiment))[0]
            df_train, df_val = df.iloc[train_idx], df.iloc[valid_idx]
            con_train = control[control.experiment.isin(df_train.experiment.unique())]
            con_val = control[control.experiment.isin(df_val.experiment.unique())]
            if opt.experiment != 'all':
                print('train_exp', df_train.experiment.unique(), 'len', df_train.shape[0])
                print('valid_exp', df_val.experiment.unique(),  'len', df_val.shape[0])
        else:
            df_train = df
            df_val = None

        img_t, img_tt = get_transform(opt.augmentation, opt.img_size)
        ds = [ImagesDataset(df_train, str(data_path), mode='train', site=[site], transforms=img_t,
                            mixup=opt.mixup, twohead=True, con_df=con_train) for site in [1, 2]]
        if df_val is not None:
            ds_val = [ImagesDataset(df_val, str(data_path), mode='train', site=[site], transforms=img_tt,
                                    twohead=True, con_df=con_val) for site in [1, 2]]
        ds_test = ImagesDataset(df_test, str(data_path), mode='test', site=[1,2], transforms=img_tt,
                                twohead=True, con_df=control_test)
        idxs = df_test.index

        loader = [D.DataLoader(ds[i], batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu) for i in range(2)]
        if df_val is not None:
            loader_val = [D.DataLoader(ds_val[i], batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu) for i in range(2)]
        loader_test = D.DataLoader(ds_test, batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu)

        trainer(0)

    else:

        # Get dataloader
        df = pd.read_csv(data_path / 'train.csv')
        df_test = pd.read_csv(data_path / 'test.csv')
        control = pd.read_csv(data_path/'train_controls.csv')
        control_test = pd.read_csv(data_path/'test_controls.csv')

        if opt.experiment != 'all': # filter exp
            df = df.loc[df.experiment.str.contains(opt.experiment)]
            df_test = df_test.loc[df_test.experiment.str.contains(opt.experiment)]
            control = control.loc[control.experiment.str.contains(opt.experiment)]
            control_test = control_test.loc[control_test.experiment.str.contains(opt.experiment)]

        exp_uni = df.experiment.unique()
        print('Experiments', exp_uni)
        if opt.cv > len(exp_uni):
            print('too many folds')
            _cv = len(exp_uni)
        else:
            _cv = opt.cv

        # gss = GroupShuffleSplit(n_splits=_cv, test_size=0.1, random_state=SEED)
        gkf = GroupKFold(n_splits=_cv)

        # for train_idx, valid_idx in gss.split(df, groups=df.experiment):
        for fold_i, (train_idx, valid_idx) in enumerate(gkf.split(df, groups=df.experiment)):
            if opt.parallel is not None and fold_i != opt.parallel:
                print(f'Parallel cross validation: skipping fold {fold_i}')
                continue

            print(f'\n-----\n starting fold {fold_i}\n-----\n')

            df_train, df_val = df.iloc[train_idx], df.iloc[valid_idx]
            con_train = control[control.experiment.isin(df_train.experiment.unique())]
            con_val = control[control.experiment.isin(df_val.experiment.unique())]

            if opt.experiment != 'all':
                print('train_exp', df_train.experiment.unique(), 'len', df_train.shape[0])
                print('valid_exp', df_val.experiment.unique(),  'len', df_val.shape[0])

            img_t, img_tt = get_transform(opt.augmentation, opt.img_size)
            ds = [ImagesDataset(df_train, str(data_path), mode='train', site=[site], transforms=img_t,
                                mixup=opt.mixup, twohead=True, con_df=con_train) for site in [1, 2]]
            if df_val is not None:
                ds_val = [ImagesDataset(df_val, str(data_path), mode='train', site=[site], transforms=img_tt,
                                        twohead=True, con_df=con_val) for site in [1, 2]]
            ds_test = ImagesDataset(df_test, str(data_path), mode='test', site=[1,2], transforms=img_tt,
                                    twohead=True, con_df=control_test)
            idxs = df_test.index

            loader = [D.DataLoader(ds[i], batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu) for i in range(2)]
            if df_val is not None:
                loader_val = [D.DataLoader(ds_val[i], batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu) for i in range(2)]
            loader_test = D.DataLoader(ds_test, batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu)

            trainer(fold_i)
