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

torch.backends.cudnn.enabled = False

'''
코드 수정 리스트

1. Weight intialization 방법 선택 (모델 레이어 내 공통 적용)
2. Dataset 선택 ( mnist, cifar100, flower, imagenet 등 다운로드 후 압축 풀도록 코드 수정 )
3. Device, gpu parallel 선택 ( single, multi, DDP )
4. report 출력 ( 그리드 형태로 예측, 예측 이미지, confusion matrix, roc-auc score, pr curve )
5. Evaluation 코드 추가


추가 기능 함수 리스트

1. Multi-Scale Crop function (Optional) 추가 -> IPYNB 구현 완료, 추후 Test 단계 적용 예정
2. Dense Evaluation function (Optional) 추가
3. VGG 전체 점진적 학습 / Transfer Learning (Optional) 추가
4. 데이터 증강 파라미터화 + 기본 데이터 증강 추가

'''

def train_model(model, criterion, optimizer, num_epochs, decay_step, num_class, dataloader, dataset_sizes, lr, decay_value, early_stop, device, save_best_state_path, model_version):
    since = time.time()

    # metrics
    acc = Accuracy()
    auroc = AUROC(task="multiclass", num_classes=num_class)
    f1 = F1Score()
    recall = Recall()
    precision = Precision()
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 10000.0
    
    # early stopping
    early_stopping_epochs = early_stop
    early_stop_counter = 0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch} / {num_epochs - 1}')
        print('-' * 10)

        # Select Phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            # running score
            running_loss, running_acc, running_auroc, running_f1, running_recall, running_precision = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

            # 데이터를 반복
            for inputs, labels in tqdm(dataloader[phase]):

                # zero the parameter gradients
                optimizer.zero_grad()
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 순전파
                outputs = model(inputs)
                if num_class == 1:
                    outputs_prob = torch.nn.functional.sigmoid(outputs)

                    # cal loss
                    loss = criterion(outputs_prob, labels)
                else:
                    outputs_prob = torch.nn.functional.softmax(outputs, dim=1)

                    # cal loss
                    loss = criterion(outputs, labels)

                # 학습 단계인 경우 역전파 + 최적화
                if phase == 'train':
                    # Backprop + optimize
                    loss.backward()
                    optimizer.step()

                # metrics
                running_loss += loss.item() * inputs.size(0)
                running_acc += acc(outputs_prob.type('torch.FloatTensor'), labels.type('torch.IntTensor'))
                running_auroc += auroc(outputs_prob.type('torch.FloatTensor'), labels.type('torch.IntTensor'))
                running_f1 += f1(outputs_prob.type('torch.FloatTensor'), labels.type('torch.IntTensor'))
                running_recall += recall(outputs_prob.type('torch.FloatTensor'), labels.type('torch.IntTensor'))
                running_precision += precision(outputs_prob.type('torch.FloatTensor'), labels.type('torch.IntTensor'))

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_acc / dataset_sizes[phase]
            epoch_auroc = running_auroc / dataset_sizes[phase]
            epoch_f1 = running_f1 / dataset_sizes[phase]
            epoch_recall = running_recall / dataset_sizes[phase]
            epoch_precision = running_precision / dataset_sizes[phase]

            print(f'{phase} Loss : {epoch_loss:.4f} Accuracy : {epoch_acc:.4f} AUROC : {epoch_auroc:.4f} F1-score : {epoch_auroc:.4f} Recall : {epoch_recall:.4f} Precision : {epoch_precision:.4f}')

            # 모델을 깊은 복사(deep copy)함
            if phase == 'valid' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                early_stop_counter = 0
                
                # best model state/model save
                save_pth_path = os.path.join(save_best_state_path, model_version) + '/best_pth.pth'
                save_pt_path = os.path.join(save_best_state_path, model_version) + '/best_pt.pt'
                torch.save(best_model_wts, save_pth_path)
                torch.save(model, save_pt_path)
                
            # 검증 데이터셋의 손실이 이전보다 증가하는 경우 / epoch 별 체크
            if phase == 'valid' and epoch_loss > best_loss:
                early_stop_counter += 1
                
        # 조기 종료 조건 확인
        if early_stop_counter >= early_stopping_epochs:
            print("Early Stopping!")
            break
                
        # Decay learning rate
        if epoch != 0 and epoch % decay_step == 0:
            lr -= decay_value
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            print ('Decay learning rate to lr: {}.'.format(lr))

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

    # 가장 나은 모델 가중치를 불러옴
    model.load_state_dict(best_model_wts)
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
    print(f"Number of parameters: {total_params}")

    # Define Dataset
    train_dataset = flowers102_dataloader.ImageFolder(config.dataset_path, config.dataset_target_path, config.split_dataset_id, config.img_size, 'train')
    valid_dataset = flowers102_dataloader.ImageFolder(config.dataset_path, config.dataset_target_path, config.split_dataset_id, config.img_size, 'valid')

    # Define Dataloader
    train_dataloader = data.DataLoader(dataset=train_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4)
    valid_dataloader = data.DataLoader(dataset=valid_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4)
    loader = {'train' : train_dataloader, 'valid' : valid_dataloader}
    dataset_sizes = {x: len(loader[x]) for x in ['train', 'valid']}
    
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
        if config.loss_function == 'CE':
            criterion = torch.nn.CrossEntropyLoss()
        elif config.loss_function == 'FOCAL':
            criterion = losses.losses.FocalLoss()

    # Training
    trained_model = train_model(model, criterion, optimizer, config.num_epochs, config.num_epochs_decay, config.num_class,
                                loader, dataset_sizes, config.lr, config.lr_decay, config.early_stopping_rounds, device, config.save_state_path, config.model)

    print('Done.')

if __name__ == '__main__':
    # parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='vgg16_advanced', help='vgg11 ~ vgg19_advanced')
    parser.add_argument('--img_size', type=int, default=112)
    parser.add_argument('--img_channels', type=int, default=3)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--num_epochs_decay', type=int, default=10)
    parser.add_argument('--early_stopping_rounds', type=int, default=10)
    parser.add_argument('--lr_decay', type=float, default=0.0001)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--optim', type=str, default='SGD')
    parser.add_argument('--num_class', type=int, default=102)
    parser.add_argument('--drop_rate', type=float, default=0.5)
    parser.add_argument('--dataset_path', type=str, default='datasets/flowers-102/jpg')
    parser.add_argument('--dataset_target_path', type=str, default='datasets/flowers-102/imagelabels.mat')
    parser.add_argument('--split_dataset_id', type=str, default='datasets/flowers-102/setid.mat')
    parser.add_argument('--save_state_path', type=str, default='runs')
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--loss_function', type=str, default='CE', help='BCE, CE, FOCAL')
    parser.add_argument('--focal_loss_gamma', type=float, default=2.0)
    parser.add_argument('--focal_loss_alpha', type=float, default=0.25)
    config = parser.parse_args()

    # main code
    main(config)