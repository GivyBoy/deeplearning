import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, num_classes=1, smooth=1.0) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, y_true, y_pred):
        return 1.0 - self._dice_coefficient(y_true, y_pred)

    def _dice_coefficient(self, y_true, y_pred):
        intersection = torch.sum(y_true * y_pred)
        union = torch.sum(y_true) + torch.sum(y_pred)
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return dice

    # def forward(self, pred, target):
    #     target = target.squeeze(1)
    #     if self.num_classes == 1:
    #         pred = torch.sigmoid(pred)
    #         pred = pred.view(-1)
    #         target = target.view(-1)
    #         intersection = (pred * target).sum()
    #         union = pred.sum() + target.sum()
    #         loss = 1 - (2.0 * intersection + self.smooth) / (union + self.smooth)
    #     else:
    #         pred = F.softmax(pred, dim=1)
    #         loss = 0
    #         for c in range(self.num_classes):
    #             pred_c = pred[:, c, :, :]
    #             target_c = (target == c).float()
    #             intersection = (pred_c * target_c).sum()
    #             union = pred_c.sum() + target_c.sum()
    #             loss += 1 - (2.0 * intersection + self.smooth) / (union + self.smooth)
    #         loss /= self.num_classes
    #     return loss
