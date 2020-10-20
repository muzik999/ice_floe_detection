import numpy as np
import torch
import torch.nn.functional as F

def dice_loss(y_true, y_pred, smooth = 1e-6):
    pred = y_pred.contiguous()
    target = y_true.contiguous()    

    intersection = (pred * target).sum(dim=2).sum(dim=2)
    
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))
    
    return loss.mean()

def soft_dice_loss(y_true, y_pred, epsilon=1e-6): 
    
    pred = y_pred.contiguous()
    target = y_true.contiguous()
    
    numerator = 2 * torch.sum(torch.mul(pred,target))
    
    denominator = torch.sum(torch.pow(pred,2)) + torch.sum(torch.pow(pred,2))
    
    return 1 - torch.mean((numerator + epsilon) / (denominator + epsilon))

def calc_loss(pred, target, weight_bce):
    pred = pred
    bce = F.binary_cross_entropy_with_logits(pred, target)
    
    pred = torch.sigmoid(pred)
    dice = dice_loss(y_true = target, y_pred = pred)
    
    loss = bce * weight_bce + dice * (1-weight_bce)
    
    return loss

def shape_aware_loss(pred, target):
    smooth = 1e-5
    
    intersect = torch.sum(pred * target)
    pred_sum = torch.sum(pred ** 2)
    gt_sum = torch.sum(target ** 2)
    L_prod = (intersect + smooth) / (intersect + pred_sum + gt_sum + smooth)
    
    sdf = L_prod + torch.norm(pred - gt_sum, 1)/torch.numel(pred)
    
    return sdf