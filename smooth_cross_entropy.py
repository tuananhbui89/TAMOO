import torch
import torch.nn as nn
import torch.nn.functional as F


def smooth_crossentropy(pred, targets, smoothing=0.1, islogits=True):
    n_class = pred.size(1)

    one_hot = torch.full_like(pred, fill_value=smoothing / (n_class - 1))
    one_hot.scatter_(dim=1, index=targets.unsqueeze(1), value=1.0 - smoothing)

    if islogits:
        log_prob = F.log_softmax(pred, dim=1)
    else: 
        log_prob = torch.log(pred)

    return F.kl_div(input=log_prob, target=one_hot, reduction='none').sum(-1)


def cross_entropy(pred, targets, smoothing=0.1, islogits=True): 

    if islogits:
        log_prob = F.log_softmax(pred, dim=1)
    else: 
        log_prob = torch.log(pred)

    n_class = pred.size(1)

    one_hot = torch.full_like(pred, fill_value=smoothing / (n_class - 1))
    one_hot.scatter_(dim=1, index=targets.unsqueeze(1), value=1.0 - smoothing)

    return -torch.sum(torch.sum(one_hot * log_prob, dim=-1)) 



