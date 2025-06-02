import os
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'  # Disable Albumentations update warning

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
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils import data
from tqdm import tqdm
from torchmetrics.classification import Accuracy, AUROC, F1Score, Recall, Precision
from sklearn.metrics import roc_curve, auc, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_recall_curve as pr_curve
from hydra import main as hydra_main
from omegaconf import DictConfig, OmegaConf
import warnings
import wandb
import gc

warnings.filterwarnings("ignore", category=UserWarning)
torch.backends.cudnn.enabled = False

def get_unique_train_folder(base_folder="runs", folder_prefix="train"):
    """만약 base_folder 내에 train 폴더가 이미 존재한다면 train2, train3 등을 반환"""
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)
    train_folder = os.path.join(base_folder, folder_prefix)
    counter = 1
    while os.path.exists(train_folder):
        train_folder = os.path.join(base_folder, f"{folder_prefix}{counter+1}")
        counter += 1
    os.makedirs(train_folder)
    return train_folder

def eval_model(model, dataloader, device, TRAIN_RESULTS_FOLDER, WEIGHTS_FOLDER):
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []  # 누적된 확률 값 저장
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            # multi-class : argmax, binary : threshold
            if outputs.shape[1] > 1:
                probs = torch.nn.functional.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)
            else:
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).long().squeeze(1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # Compute and save confusion matrix
    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    cm_path = os.path.join(TRAIN_RESULTS_FOLDER, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()
    print(f"Confusion Matrix saved at: {cm_path}")
    
    # --- ROC and PR curves ---
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)
    num_unique = np.unique(all_targets).size
    if num_unique > 2:
        # Multi-class ROC/PR: Binarize ground truth
        classes = np.unique(all_targets)
        n_classes = len(classes)
        y_test = label_binarize(all_targets, classes=classes)  # shape: (n_samples, n_classes)
        # all_probs shape: (n_samples, n_classes) is assumed

        # Micro-average ROC
        fpr_micro, tpr_micro, _ = roc_curve(y_test.ravel(), all_probs.ravel())
        roc_auc_micro = auc(fpr_micro, tpr_micro)
        # Per-class ROC and macro-average ROC
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test[:, i], all_probs[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= n_classes
        fpr_macro = all_fpr
        tpr_macro = mean_tpr
        roc_auc_macro = auc(fpr_macro, tpr_macro)
        
        # Plot ROC curves
        plt.figure()
        plt.plot(fpr_micro, tpr_micro, color='deeppink', linestyle=':', linewidth=4,
                 label='Micro-average ROC (AUC = {0:0.2f})'.format(roc_auc_micro))
        plt.plot(fpr_macro, tpr_macro, color='navy', linestyle='-', linewidth=4,
                 label='Macro-average ROC (AUC = {0:0.2f})'.format(roc_auc_macro))
        plt.plot([0,1],[0,1],'k--',lw=2)
        plt.xlim([0.0,1.0])
        plt.ylim([0.0,1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves (Multi-class)')
        plt.legend(loc="lower right")
        roc_path = os.path.join(TRAIN_RESULTS_FOLDER, "roc_curves.png")
        plt.savefig(roc_path)
        plt.close()
        print(f"ROC curves saved at: {roc_path}")
        
        # Micro-average PR
        precision_micro, recall_micro, _ = pr_curve(y_test.ravel(), all_probs.ravel())
        pr_auc_micro = auc(recall_micro, precision_micro)
        # Per-class PR and macro-average PR
        precision_dict = dict()
        recall_dict = dict()
        for i in range(n_classes):
            precision_dict[i], recall_dict[i], _ = pr_curve(y_test[:, i], all_probs[:, i])
        all_recall = np.unique(np.concatenate([recall_dict[i] for i in range(n_classes)]))
        mean_precision = np.zeros_like(all_recall)
        for i in range(n_classes):
            mean_precision += np.interp(all_recall, recall_dict[i], precision_dict[i])
        mean_precision /= n_classes
        pr_auc_macro = auc(all_recall, mean_precision)
        
        # Plot PR curves
        plt.figure()
        plt.plot(recall_micro, precision_micro, color='green', linestyle=':', linewidth=4,
                 label='Micro-average PR (AUC = {0:0.2f})'.format(pr_auc_micro))
        plt.plot(all_recall, mean_precision, color='blue', linestyle='-', linewidth=4,
                 label='Macro-average PR (AUC = {0:0.2f})'.format(pr_auc_macro))
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves (Multi-class)')
        plt.legend(loc="lower left")
        pr_multi_path = os.path.join(TRAIN_RESULTS_FOLDER, "pr_curves_multiclass.png")
        plt.savefig(pr_multi_path)
        plt.close()
        print(f"Multi-class PR curves saved at: {pr_multi_path}")
    else:
        # Binary classification: 기존 precision-recall curve already exists
        precision_vals, recall_vals, _ = pr_curve(all_targets, all_probs.ravel())
        plt.figure()
        plt.plot(recall_vals, precision_vals, marker='.', label='PR curve')
        plt.title('Precision-Recall Curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        pr_path = os.path.join(TRAIN_RESULTS_FOLDER, "precision_recall_curve.png")
        plt.savefig(pr_path)
        plt.close()
        print(f"Precision-Recall curve saved at: {pr_path}")
        
        # Generate threshold plots for binary classification
        thresholds = np.linspace(0, 1, 100)
        f1_list = []
        precision_list = []
        recall_list = []
        # all_probs.ravel() is assumed to be probability of positive class
        binary_probs = all_probs.ravel()
        for t in thresholds:
            pred_thresh = (binary_probs >= t).astype(int)
            f1_list.append(f1_score(all_targets, pred_thresh))
            precision_list.append(precision_score(all_targets, pred_thresh))
            recall_list.append(recall_score(all_targets, pred_thresh))
            
        # Plot F1 vs. Threshold
        plt.figure()
        plt.plot(thresholds, f1_list, label="F1 score")
        plt.xlabel("Threshold")
        plt.ylabel("F1 score")
        plt.title("F1 Score vs Threshold")
        plt.legend()
        f1_thresh_path = os.path.join(TRAIN_RESULTS_FOLDER, "f1_vs_threshold.png")
        plt.savefig(f1_thresh_path)
        plt.close()
        print(f"F1 vs Threshold curve saved at: {f1_thresh_path}")
        
        # Plot Precision vs. Threshold
        plt.figure()
        plt.plot(thresholds, precision_list, label="Precision")
        plt.xlabel("Threshold")
        plt.ylabel("Precision")
        plt.title("Precision vs Threshold")
        plt.legend()
        precision_thresh_path = os.path.join(TRAIN_RESULTS_FOLDER, "precision_vs_threshold.png")
        plt.savefig(precision_thresh_path)
        plt.close()
        print(f"Precision vs Threshold curve saved at: {precision_thresh_path}")
        
        # Plot Recall vs. Threshold
        plt.figure()
        plt.plot(thresholds, recall_list, label="Recall")
        plt.xlabel("Threshold")
        plt.ylabel("Recall")
        plt.title("Recall vs Threshold")
        plt.legend()
        recall_thresh_path = os.path.join(TRAIN_RESULTS_FOLDER, "recall_vs_threshold.png")
        plt.savefig(recall_thresh_path)
        plt.close()
        print(f"Recall vs Threshold curve saved at: {recall_thresh_path}")
    
    # Save current model weights in WEIGHTS_FOLDER (pth & pt)
    pth_path = os.path.join(WEIGHTS_FOLDER, "model_best.pth")
    pt_path = os.path.join(WEIGHTS_FOLDER, "model_best.pt")
    print(os.getcwd())
    print(f"Absolute WEIGHTS_FOLDER: {os.path.abspath(WEIGHTS_FOLDER)}")
    torch.save(model.state_dict(), pth_path)
    torch.save(model, pt_path)
    print(f"Model weights saved: {pth_path} and {pt_path}")
    
    return None

def train_model(model, criterion, optimizer, num_epochs, decay_step, num_class, dataloader, dataset_sizes,
                lr, decay_value, early_stop, device, save_best_state_path, model_version,
                TRAIN_RESULTS_FOLDER, WEIGHTS_FOLDER):
    since = time.time()

    # metrics
    acc = Accuracy(task="multiclass", average='micro', num_classes=num_class, top_k=1).to(device)
    auroc = AUROC(task="multiclass", num_classes=num_class).to(device)
    f1 = F1Score(task="multiclass", average='micro', num_classes=num_class, top_k=1).to(device)
    recall = Recall(task="multiclass", average='micro', num_classes=num_class, top_k=1).to(device)
    precision = Precision(task="multiclass", average='micro', num_classes=num_class, top_k=1).to(device)
    
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
            # choose phase  
            if phase == 'train':
                model.train()
            else:
                model.eval()

            # running score
            running_loss, running_acc, running_auroc, running_f1, running_recall, running_precision = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

            # 데이터를 반복적으로 불러오기
            if phase == 'train':
                for inputs, labels in tqdm(dataloader[phase]):
                    optimizer.zero_grad()
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    outputs = model(inputs)
                    if num_class == 1:
                        outputs_prob = torch.nn.functional.sigmoid(outputs)
                        loss = criterion(outputs, labels)
                    else:
                        outputs_prob = torch.nn.functional.softmax(outputs, dim=1)
                        loss = criterion(outputs, labels)
                    
                    loss.backward()
                    optimizer.step()

                    # Loss 
                    running_loss += loss.item() * inputs.size(0)
                    
                    # Metrics
                    acc.update(torch.argmax(outputs_prob.float(), dim=-1), labels.int())
                    auroc.update(outputs_prob.float(), labels.int())
                    f1.update(torch.argmax(outputs_prob.float(), dim=-1), labels.int())
                    recall.update(torch.argmax(outputs_prob.float(), dim=-1), labels.int())
                    precision.update(torch.argmax(outputs_prob.float(), dim=-1), labels.int())
            else :
                with torch.no_grad():
                    for inputs, labels in tqdm(dataloader[phase]):
                        optimizer.zero_grad()
                        inputs = inputs.to(device)
                        labels = labels.to(device)

                        outputs = model(inputs)
                        if num_class == 1:
                            outputs_prob = torch.nn.functional.sigmoid(outputs)
                            loss = criterion(outputs, labels)
                        else:
                            outputs_prob = torch.nn.functional.softmax(outputs, dim=1)
                            loss = criterion(outputs, labels)

                        # Loss 
                        running_loss += loss.item() * inputs.size(0)
                        
                        # Metrics
                        acc.update(torch.argmax(outputs_prob.float(), dim=-1), labels.int())
                        auroc.update(outputs_prob.float(), labels.int())
                        f1.update(torch.argmax(outputs_prob.float(), dim=-1), labels.int())
                        recall.update(torch.argmax(outputs_prob.float(), dim=-1), labels.int())
                        precision.update(torch.argmax(outputs_prob.float(), dim=-1), labels.int())

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = acc.compute()
            epoch_auroc = auroc.compute()
            epoch_f1 = f1.compute()
            epoch_recall = recall.compute()
            epoch_precision = precision.compute()

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
                    
                    # 모델 저장 (WEIGHTS_FOLDER에 저장)
                    save_pth_path = os.path.join(WEIGHTS_FOLDER, "training_best.pth")
                    save_pt_path = os.path.join(WEIGHTS_FOLDER, "training_best.pt")
                    torch.save(best_model_wts, save_pth_path)
                    torch.save(model, save_pt_path)
                else:
                    early_stop_counter += 1
        
            # reset metrics for next epoch
            acc.reset()
            auroc.reset()
            f1.reset()
            recall.reset()
            precision.reset()
            
            # Garbage collection to free up memory every epoch
            gc.collect()
            torch.cuda.empty_cache()
        
        # Log metrics to wandb
        wandb.log({
            "epoch": epoch,
            "train_loss": epoch_train_metrics.get("loss"),
            "valid_loss": epoch_valid_metrics.get("loss"),
            "train_accuracy": epoch_train_metrics.get("accuracy"),
            "valid_accuracy": epoch_valid_metrics.get("accuracy"),
            "train_auroc": epoch_train_metrics.get("auroc"),
            "valid_auroc": epoch_valid_metrics.get("auroc"),
            "train_f1": epoch_train_metrics.get("f1"),
            "valid_f1": epoch_valid_metrics.get("f1")
        })
        
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
    
    # 검증 데이터를 다시 평가하여 시각적 자료 저장 (eval_model 호출)
    print("Save results... : {}".format(TRAIN_RESULTS_FOLDER))
    print("Save Models... : {}".format(WEIGHTS_FOLDER))
    eval_model(model, dataloader['valid'], device, TRAIN_RESULTS_FOLDER, WEIGHTS_FOLDER)
    
    return model

def main(config):
    TRAIN_RESULTS_FOLDER = get_unique_train_folder()  
    WEIGHTS_FOLDER = os.path.join(TRAIN_RESULTS_FOLDER, "weights")
    os.makedirs(WEIGHTS_FOLDER, exist_ok=True)
    
    # Initialize wandb with your project name and config
    wandb.init(project="VisionBaseline", config=OmegaConf.to_container(config, resolve=True))
    
    # Define Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # # Define Network
    # if 'vgg' in config.model:
    #     model = VGG.vgg(config.num_class,
    #                    (config.img_channels, config.img_size, config.img_size),
    #                     config.drop_rate, config.model)
    
    # Testing with torchvision's VGG model
    model = torchvision.models.vgg16(pretrained=True)
    model.classifier[6] = nn.Linear(in_features=4096, out_features=config.num_class)
    
    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Number of trainable parameters: {total_params}")
    
    # Optionally, watch the model
    wandb.watch(model)

    # Define Dataset
    print('Loading dataset...')
    train_dataset = flowers102_dataloader.ImageFolder(config.dataset_path, config.dataset_target_path,
                                                      config.split_dataset_id, config.img_size, 'train')
    valid_dataset = flowers102_dataloader.ImageFolder(config.dataset_path, config.dataset_target_path,
                                                      config.split_dataset_id, config.img_size, 'valid')

    # Define Dataloader
    train_dataloader = data.DataLoader(dataset=train_dataset, batch_size=config.batch_size,
                                       shuffle=True, num_workers=4, pin_memory=True)
    valid_dataloader = data.DataLoader(dataset=valid_dataset, batch_size=config.batch_size,
                                       shuffle=False, num_workers=4, pin_memory=True)
    
    loader = {'train' : train_dataloader, 'valid' : valid_dataloader}
    dataset_sizes = {x: len(loader[x]) for x in ['train', 'valid']}
    print(f"Train dataset size: {dataset_sizes['train']}")
    print(f"Valid dataset size: {dataset_sizes['valid']}")
    
    print('Set Optimizer & Loss function...')
    # Define Optimizer & Loss function
    optimizer = None
    if config.optim == 'SGD':
        optimizer = torch.optim.SGD(list(model.parameters()), config.lr, 0.9, weight_decay=config.weight_decay)
    elif config.optim == 'Adam':
        optimizer = torch.optim.Adam(list(model.parameters()), config.lr, [0.9, 0.999], weight_decay=config.weight_decay)
    elif config.optim == 'AdamW':
        optimizer = torch.optim.AdamW(list(model.parameters()), config.lr, [0.9, 0.999], weight_decay=config.weight_decay)

    criterion = None
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
    print(f'Loss function: {config.loss_function}, Optimizer: {config.optim}')

    # Training
    trained_model = train_model(model, criterion, optimizer, config.num_epochs, config.num_epochs_decay,
                                config.num_class, loader, dataset_sizes, config.lr, config.lr_decay,
                                config.early_stopping_rounds, device, config.save_state_path, config.model,
                                TRAIN_RESULTS_FOLDER, WEIGHTS_FOLDER)

    print('Done.')

@hydra_main(config_path=".", config_name="config")
def hydra_main_function(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    main(cfg)

if __name__ == '__main__':
    hydra_main_function()