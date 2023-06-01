
from multiprocessing.sharedctypes import Value
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 

class Ensemble(nn.Module):
    def __init__(self, models):
        super(Ensemble, self).__init__()
        self.models = models
        assert len(self.models) > 0

    def forward(self, x):
        if len(self.models) > 1:
            outputs = 0
            for model in self.models:
                outputs += F.softmax(model(x), dim=-1)
            output = outputs / len(self.models)
            output = torch.clamp(output, min=1e-40)
            return torch.log(output)
        else:
            return self.models[0](x)

def weighted_ensemble(models, x, weights, enmechan='prob'):
    """
    weighted prediction of an ensemble 
    Args: 
        models: list of models 
        x: input, [batch_size, C, H, W]
        weights: weight of each model 
            support two formats: [num_models] or [batch_size, num_models]
        enmechan: ensemble mechanism 
            prob: average probability vectors (after softmax)
            logit: average logit vectors
    Return: 
        ensembled logit vectors [batch_size, num_classes] 
    """ 
    num_models = len(models)
    batch_size = x.shape[0]
    assert(num_models == weights.shape[-1])

    if len(weights.shape) == 1: 
        batchw = torch.unsqueeze(weights, dim=0) 
        batchw = torch.repeat_interleave(batchw, repeats=batch_size, dim=0)
    elif len(weights.shape) == 2: 
        batchw = weights
    else: 
        raise ValueError
    
    if enmechan == 'prob':
        _outputs = 0 
        for im, _model in enumerate(models): 
            _outputs += torch.unsqueeze(batchw[:,im], dim=1) * F.softmax(_model(x), dim=-1)
        
        output = torch.clamp(_outputs, min=1e-40)
        return torch.log(output)
    elif enmechan == 'logit': 
        _outputs = 0 
        for im, _model in enumerate(models): 
            _outputs += torch.unsqueeze(batchw[:,im], dim=1) * _model(x)

        return _outputs

def test_weighted_ensemble(): 
    X = torch.ones(size=[3,5])
    w = torch.tensor(np.asarray([1,2,3,4,5]))
    w = torch.unsqueeze(w, dim=0)
    w_rp = torch.repeat_interleave(w, repeats=3, dim=0)

    print(torch.unsqueeze(w_rp[:,0], dim=1) * X) 
    print(torch.unsqueeze(w_rp[:,1], dim=1) * X)
    print(torch.unsqueeze(w_rp[:,2], dim=1) * X)

if __name__ == '__main__': 
    test_weighted_ensemble()