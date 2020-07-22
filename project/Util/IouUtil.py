import torch
import numpy as np

SMOOTH = 1e-6

def class_acc(outputs: torch.Tensor, labels: torch.Tensor, device='cuda'):
    outputs, labels = outputs.argmax(dim=1), labels.flatten()
    return (outputs==labels).to(torch.float32).mean()

def iou_numpy(outputs: np.array, labels: np.array):
    outputs = outputs.squeeze(1)

    intersection = (outputs & labels).sum((1, 2))
    union = (outputs | labels).sum((1, 2))

    iou = (intersection + SMOOTH) / (union + SMOOTH)

    thresholded = np.ceil(np.clip(20 * (iou - 0.5), 0, 10)) / 10

    return thresholded  # Or thresholded.mean()


def iou_pytorch(outputs: torch.Tensor, labels: torch.Tensor, device='cuda'):
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from TF or something it will most probably
    # be with the BATCH x 1 x H x W shape
    # outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W
    outputs = torch.max(outputs, 1)[1]
    batch_size = outputs.size()[0]
    # intersection = torch.where(((outputs.int() != 0) | (labels.int() != 0)) & (outputs.int() == labels.int()), torch.Tensor([1]).to(device), torch.Tensor([0]).to(device))
    # intersection = intersection.float().sum((1, 2))
    # intersection = intersection.float().sum()
    # intersection = (outputs.int() & labels.int()).float().sum((1, 2))
    intersection = ((outputs.int() != 0) & (labels.int() != 0) & (outputs.int() == labels.int())).float()
    intersection = intersection.view(batch_size, -1).sum(1)

    # union = torch.where(((outputs.int() != 0) | (labels.int() != 0)), torch.Tensor([1]).to(device), torch.Tensor([0]).to(device))
    # union = union.float().sum((1, 2))
    # union = union.float().sum()
    # union = (outputs.int() | labels.int()).float().sum((1, 2))
    union = ((outputs.int() != 0) | (labels.int() != 0)).float()
    union = union.view(batch_size, -1).sum(1)

    iou = (intersection + SMOOTH) / (union + SMOOTH)

    return iou.mean()
#     iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0

#     thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds

#     return thresholded.mean()  # Or thresholded.mean() if you are interested in average across the batch
