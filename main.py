import os
import copy
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import argparse
import dataloaders.flowers102_dataloader as flowers102_dataloader
import losses.losses
import networks.VGG as VGG
import losses
from torch.utils import data
from tqdm import tqdm
from torchmetrics.classification import Accuracy, AUROC, F1Score, Recall, Precision
from sklearn.metrics import classification_report, confusion_matrix
from hydra import main as hydra_main
from omegaconf import DictConfig, OmegaConf

torch.backends.cudnn.enabled = False

def eval_model(model, dataloader, device):
    
    
    
    
    
    return None

def train_model(model, criterion, optimizer, num_epochs, decay_step, num_class, dataloader, dataset_sizes, lr, decay_value, early_stop, device, save_best_state_path, model_version):
    since = time.time()

    # metrics
    acc = Accuracy()
    auroc = AUROC(task="multiclass", num_classes=num_class)
    f1 = F1Score()
    recall = Recall()
    precision = Precision()
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10  # 매우 큰 값으로 초기화
    
    # early stopping
    early_stopping_epochs = early_stop
    early_stop_counter = 0

    # best 정보를 저장할 변수
    best_epoch = -1
    best_train_metrics = {}
    best_valid_metrics = {}

    for epoch in range(num_epochs):
        print(f'Epoch {epoch} / {num_epochs - 1}')
        print('-' * 10)

        # 임시로 epoch마다의 metric 저장용 변수
        epoch_train_metrics = {}
        epoch_valid_metrics = {}

        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            # running score
            running_loss, running_acc, running_auroc, running_f1, running_recall, running_precision = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

            # 데이터를 반복
            for inputs, labels in tqdm(dataloader[phase]):
                optimizer.zero_grad()
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                if num_class == 1:
                    outputs_prob = torch.nn.functional.sigmoid(outputs)
                    loss = criterion(outputs_prob, labels)
                else:
                    outputs_prob = torch.nn.functional.softmax(outputs, dim=1)
                    loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_acc += acc(outputs_prob.float(), labels.int())
                running_auroc += auroc(outputs_prob.float(), labels.int())
                running_f1 += f1(outputs_prob.float(), labels.int())
                running_recall += recall(outputs_prob.float(), labels.int())
                running_precision += precision(outputs_prob.float(), labels.int())

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_acc / dataset_sizes[phase]
            epoch_auroc = running_auroc / dataset_sizes[phase]
            epoch_f1 = running_f1 / dataset_sizes[phase]
            epoch_recall = running_recall / dataset_sizes[phase]
            epoch_precision = running_precision / dataset_sizes[phase]

            print(f'{phase} Loss : {epoch_loss:.4f} Accuracy : {epoch_acc:.4f} AUROC : {epoch_auroc:.4f} F1-score : {epoch_f1:.4f} Recall : {epoch_recall:.4f} Precision : {epoch_precision:.4f}')

            if phase == 'train':
                epoch_train_metrics = {
                    'loss': epoch_loss,
                    'accuracy': epoch_acc,
                    'auroc': epoch_auroc,
                    'f1': epoch_f1,
                    'recall': epoch_recall,
                    'precision': epoch_precision
                }
            else:  # valid phase
                epoch_valid_metrics = {
                    'loss': epoch_loss,
                    'accuracy': epoch_acc,
                    'auroc': epoch_auroc,
                    'f1': epoch_f1,
                    'recall': epoch_recall,
                    'precision': epoch_precision
                }
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    early_stop_counter = 0
                    best_epoch = epoch
                    best_train_metrics = epoch_train_metrics.copy()
                    best_valid_metrics = epoch_valid_metrics.copy()
                    
                    # 모델 저장
                    save_pth_path = os.path.join(save_best_state_path, model_version) + '/best_pth.pth'
                    save_pt_path = os.path.join(save_best_state_path, model_version) + '/best_pt.pt'
                    torch.save(best_model_wts, save_pth_path)
                    torch.save(model, save_pt_path)
                else:
                    early_stop_counter += 1

        if early_stop_counter >= early_stopping_epochs:
            print("Early Stopping!")
            break
                
        if epoch != 0 and epoch % decay_step == 0:
            lr -= decay_value
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            print('Decay learning rate to lr: {}.'.format(lr))
        print()

    time_elapsed = time.time() - since
    print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

    # 모델 학습이 종료된 후 best epoch와 해당 지표들을 출력
    print("\nBest model found at epoch {}:".format(best_epoch))
    print("Training Metrics at best epoch:")
    print("  Loss     : {:.4f}".format(best_train_metrics.get('loss', 0)))
    print("  Accuracy : {:.4f}".format(best_train_metrics.get('accuracy', 0)))
    print("  AUROC    : {:.4f}".format(best_train_metrics.get('auroc', 0)))
    print("  F1-score : {:.4f}".format(best_train_metrics.get('f1', 0)))
    print("  Recall   : {:.4f}".format(best_train_metrics.get('recall', 0)))
    print("  Precision: {:.4f}".format(best_train_metrics.get('precision', 0)))

    print("Validation Metrics at best epoch:")
    print("  Loss     : {:.4f}".format(best_valid_metrics.get('loss', 0)))
    print("  Accuracy : {:.4f}".format(best_valid_metrics.get('accuracy', 0)))
    print("  AUROC    : {:.4f}".format(best_valid_metrics.get('auroc', 0)))
    print("  F1-score : {:.4f}".format(best_valid_metrics.get('f1', 0)))
    print("  Recall   : {:.4f}".format(best_valid_metrics.get('recall', 0)))
    print("  Precision: {:.4f}".format(best_valid_metrics.get('precision', 0)))

    # 가장 나은 모델 가중치 불러오기
    model.load_state_dict(best_model_wts)
    
    # 학습 모델 기반 성능 지표 도출 및 저장
    eval_model(model, dataloader['valid'], device)
    
    return model

def main(config):
    # Define Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define Network
    if 'vgg' in config.model:
        model = VGG.vgg(config.num_class,
                       (config.img_channels, config.img_size, config.img_size),
                        config.drop_rate, config.model)
    
    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Number of trainable parameters: {total_params}")

    # Define Dataset
    print('Loading dataset...')
    train_dataset = flowers102_dataloader.ImageFolder(config.dataset_path, config.dataset_target_path,
                                                      config.split_dataset_id, config.img_size, 'train')
    valid_dataset = flowers102_dataloader.ImageFolder(config.dataset_path, config.dataset_target_path,
                                                      config.split_dataset_id, config.img_size, 'valid')

    # Define Dataloader
    train_dataloader = data.DataLoader(dataset=train_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4)
    valid_dataloader = data.DataLoader(dataset=valid_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4)
    
    loader = {'train' : train_dataloader, 'valid' : valid_dataloader}
    dataset_sizes = {x: len(loader[x]) for x in ['train', 'valid']}
    print(f"Train dataset size: {dataset_sizes['train']}")
    print(f"Valid dataset size: {dataset_sizes['valid']}")
    
    print('Set Optimizer & Loss function...')
    # Define Optimizer & Loss function
    if config.optim == 'SGD':
        optimizer = torch.optim.SGD(list(model.parameters()), config.lr, 0.9, weight_decay=config.weight_decay)
    elif config.optim == 'Adam':
        optimizer = torch.optim.Adam(list(model.parameters()), config.lr, [0.9, 0.999], weight_decay=config.weight_decay)
    elif config.optim == 'AdamW':
        optimizer = torch.optim.AdamW(list(model.parameters()), config.lr, [0.9, 0.999], weight_decay=config.weight_decay)

    if config.num_class == 1:
        if config.loss_function == 'BCE':
            criterion = torch.nn.BCEWithLogitsLoss()
        elif config.loss_function == 'FOCAL':
            criterion = losses.losses.BinaryFocalLoss()
    else:
        if config.loss_function == 'CrossEntropy':
            criterion = torch.nn.CrossEntropyLoss()
        elif config.loss_function == 'FOCAL':
            criterion = losses.losses.FocalLoss()
    print(f'Loss function: {config.loss_function}, Optimizer: {config.optim}')

    # Training
    trained_model = train_model(model, criterion, optimizer, config.num_epochs, config.num_epochs_decay,
                                config.num_class, loader, dataset_sizes, config.lr, config.lr_decay,
                                config.early_stopping_rounds, device, config.save_state_path, config.model)

    print('Done.')

@hydra_main(config_path=".", config_name="config")
def hydra_main_function(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    main(cfg)

if __name__ == '__main__':
    hydra_main_function()