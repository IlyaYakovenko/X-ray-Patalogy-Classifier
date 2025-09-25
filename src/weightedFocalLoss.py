import torch.nn as nn
import torch
import torch.nn.functional as F

class WeightedFocalLoss(nn.Module):
    def __init__(self, class_weights, alpha=1, gamma=2, reduction='mean'):

        super(WeightedFocalLoss, self).__init__()
        self.class_weights = class_weights
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):



        probs = torch.sigmoid(inputs)


        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none'
        )


        focal_weights = torch.where(
            targets == 1,
            (1 - probs).pow(self.gamma),
            probs.pow(self.gamma)
        )


        weight_matrix = torch.where(
            targets == 1,
            self.class_weights.unsqueeze(0),
            torch.ones_like(inputs)
        )


        focal_loss = self.alpha * weight_matrix * focal_weights * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss
