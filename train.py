import os
import argparse

import yaml
import torch

from models import Model
from utils import loss_func, optim_func, lr_scheduler_func, plot_progress, save_model, get_train_dataset, get_dataloader

def train(args, cfg):
    '''
    Train model
    '''
    ########################
    #   Get configuration  #
    ########################
    device = torch.device('cuda' if cfg['device']=='cuda' and torch.cuda.is_available() else 'cpu')
    train_cfg = cfg['train']
    batch_size = train_cfg['batch_size']
    train_epochs = train_cfg['train_epochs']
    loss_name_list = train_cfg['loss']
    optim_cfg = train_cfg['optim']
    lr_scheduler_cfg = train_cfg['lr_scheduler']
    model_path = train_cfg['model_path']
    progress_path = train_cfg['progress_path']
    plot_epochs = train_cfg['plot_epochs']
    model_cfg = cfg['model']

    ########################
    #      Make model      #
    ########################
    model = Model(model_cfg).to(device)

    ########################
    #    train settings    #
    ########################
    train_dataset, valid_dataset = get_train_dataset()
    train_loader, valid_loader = get_dataloader([train_dataset, valid_dataset], batch_size)
    criterion_list = loss_func(loss_name_list)
    optimizer = optim_func(model, optim_cfg)
    lr_scheduler = lr_scheduler_func(optimizer, lr_scheduler_cfg)
    history = {'train':[], 'validation':[]} # for saving loss
    start_epoch = 1

    os.makedirs(model_path, exist_ok=True)
    os.makedirs(progress_path, exist_ok=True)

    if args.resume: # if pretrained model exists
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch']+1
        model.load_state_dict(checkpoint['model_state_dict'])
        history = checkpoint['history']
        optimizer.load_state_dict(checkpoint['optim_state_dict'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])


    ########################
    #      Train model     #
    ########################
    for epoch in range(start_epoch, train_epochs+1):
        total_loss = 0.0
        for input, target in train_loader:
            input, target = input.to(device), target.to(device)
            output = model(input)
            optimizer.zero_grad()
            loss = 0.0
            for criterion in criterion_list:
                loss += criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        total_loss /= len(train_loader)
        history['train'].append(total_loss)
        validation_loss = validation(model, valid_loader, criterion_list, device)
        history['valid'].append(validation_loss)
        print(f'------ {epoch:03d} training ------- train loss : {total_loss:.6f} -------- validation loss : {validation_loss:.6f}-------')
        if epoch % plot_epochs == 0:
            plot_progress(history, epoch, progress_path)
            save_model(model, epoch, model_path, optimizer, lr_scheduler, history)
        lr_scheduler.step()

def validation(model, validation_loader, criterion, device):
    total_loss = 0.0
    for input, _ in validation_loader:
        input = input.squeeze(dim=1).to(device)
        target = input.detach()
        with torch.no_grad():
            output = model(input).detach()
        loss = criterion(output, target)
        total_loss += loss.item()
    total_loss /= len(validation_loader)
    return total_loss

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/config.yaml', help='Path to config file')
    parser.add_argument('--resume', type=str, default='', help='Path to pretrained model file')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    train(args, cfg)