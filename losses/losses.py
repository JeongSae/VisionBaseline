import torch
import torch.nn as nn
import torch.nn.functional as F

class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Sigmoid를 이용해 logits을 확률로 변환
        inputs = torch.sigmoid(inputs)
        
        # Positive 및 Negative 확률
        p_t = inputs * targets + (1 - inputs) * (1 - targets)

        # Focal Loss 계산
        focal_loss = -self.alpha * (1 - p_t) ** self.gamma * torch.log(p_t + 1e-8)  # 작은 값 1e-8 추가는 log의 안정성을 위해
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
        
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Cross Entropy Loss 계산
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')

        # p_t 계산 (정답 클래스에 대한 확률)
        p_t = torch.exp(-ce_loss)

        # Focal Loss 계산
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss