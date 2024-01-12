import torch
from torch import Tensor

def pearson_correlation_batch_cuda_0(batch_x, batch_y, dim=1, eps=1e-6):
    n = batch_x.shape[dim]
    sum_x = torch.sum(batch_x, dim=dim)
    sum_y = torch.sum(batch_y, dim=dim)
    sum_xy = torch.sum(batch_x * batch_y, dim=dim)
    sum_x2 = torch.sum(batch_x ** 2, dim=dim)
    sum_y2 = torch.sum(batch_y ** 2, dim=dim)

    numerator = n * sum_xy - sum_x * sum_y
    denominator = torch.sqrt((n * sum_x2 - sum_x ** 2) * (n * sum_y2 - sum_y ** 2) + eps)

    correlations = numerator / denominator
    return correlations

def pearson_correlation_batch_cuda_1(batch_x, batch_y, dim=1, eps=1e-6):
    mean_x = torch.mean(batch_x, dim=dim, keepdim=True)
    mean_y = torch.mean(batch_y, dim=dim, keepdim=True)
    mean_xy = (batch_x - mean_x) * (batch_y - mean_y)
    mean_xy = torch.mean(mean_xy, dim=dim)
    numerator = mean_xy
    denominator = torch.sqrt(torch.var(batch_x, dim=dim, unbiased=False) * torch.var(batch_y, dim=dim, unbiased=False))

    correlations = numerator / denominator
    return correlations

def pearson_correlation_batch_cuda(x: Tensor, y: Tensor, dim: int = -1, eps: float = 1.e-6) -> Tensor:
    x_norm = (x - torch.mean(x, dim=dim, keepdim=True)) / torch.std(x, dim=dim, unbiased=False, keepdim=True)
    y_norm = (y - torch.mean(y, dim=dim, keepdim=True)) / torch.std(y, dim=dim, unbiased=False, keepdim=True)
    corr = torch.mean(x_norm * y_norm, dim=-1)
    return corr


