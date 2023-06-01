import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

LARGE_CONST = 1e9 

def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)

def random_select_target(true_labels, num_classes): 
    """
    Return the non-true-label target for pgd targeted attack
    Args: 
        true_labels: the true labels of inputs [b, ], categorical format  
        num_classes: number of classes  
    """
    device = true_labels.get_device() if true_labels.get_device() >= 0 else "cpu"
    one_hot = F.one_hot(true_labels, num_classes).to(device=device)
    ran_vec = torch.rand(size=one_hot.shape, device=device)
    sub_vec = ran_vec - 2*one_hot 
    new_labels = torch.argmax(sub_vec, dim=-1).detach()
    return new_labels 

def test_random_select_target(): 
    true_labels = [0, 5, 6, 7, 8]
    true_labels = torch.tensor(true_labels)
    new_labels = random_select_target(true_labels, num_classes=10)
    print(new_labels)

def switch_status(model, status): 
    if status == 'train': 
        model.train()
    elif status == 'eval': 
        model.eval()
    else: 
        raise ValueError

def Cosine(g1, g2):
	return torch.abs(F.cosine_similarity(g1, g2)).mean()  # + (0.05 * torch.sum(g1**2+g2**2,1)).mean()

def Magnitude(g1):
	return (torch.sum(g1**2,1)).mean() * 2

def cw_loss(target, logits, confidence=50, reduction='sum'):
    """
    Note that: 
        MinMax paper: cw_loss = max(correct_logit - wrong_logit + confidence, 0). Attack minimize this loss 
        Org C&W paper: cw_loss = max(wrong_logit - correct_logit + confidence, 0). Attack maximize this loss 
        This paper == Org C&W paper. Attack maximize this loss  
    """

    assert(len(target.shape) == 1)
    num_classes = logits.shape[1]
    target_onehot = F.one_hot(target, num_classes=num_classes) 
    correct_logit = torch.sum(target_onehot * logits, dim=1) # [B,]
    wrong_logit, _ = torch.max(target_onehot * logits - 1e4 * target_onehot, dim=1) # [B,]
    loss = torch.relu(wrong_logit - correct_logit + confidence) # [B,]
    if reduction == 'sum': 
        return torch.sum(loss, dim=0)
    elif reduction == 'mean': 
        return torch.mean(loss, dim=0)
    elif reduction == 'none': 
        return loss 

def wrap_loss_fn(target, adv_logits, nat_logits, reduction='sum', loss_type='ce'): 
    assert(len(target.shape) == 1)
    if loss_type == 'ce': 
        # return F.cross_entropy(adv_logits, target, reduction=reduction, label_smoothing=0.01)
        return F.cross_entropy(adv_logits, target, reduction=reduction)
    elif loss_type == 'kl': 
        temp = F.kl_div(F.log_softmax(adv_logits, dim=1), 
									F.softmax(nat_logits, dim=1), 
									reduction='none') # [B, num_classes]
        assert(len(temp.shape)==2)
        if reduction == 'sum': 
            return torch.sum(torch.sum(temp, dim=1), dim=0)
        elif reduction == 'mean': 
            return torch.mean(torch.sum(temp, dim=1), dim=0)
        elif reduction == 'none': 
            return torch.sum(temp, dim=1)

    elif loss_type == 'cw': 
        return cw_loss(target, adv_logits, reduction=reduction)
    
def gen_mask(logits, target): 
    """
    Generate mask regarding current prediction probability, output shape: [batch_size,]
    Returns: 
        mask =  (pred == target, 0.0, 1.0)
        pred == target --> correct prediction (unsuccesful attack) --> return 0 
        pred != target --> incorrect prediction (successful attack) --> return 1 
    """
    assert(len(logits.shape) == 2)
    assert(len(target.shape) == 1)
    pred = torch.argmax(logits, dim=1)
    return torch.where(pred == target, 0.0, 1.0)

def max_non_target(logits, target): 
    """
    return value and index of maximum non-true value
    """
    assert(len(logits.shape) == 2)
    assert(len(target.shape) == 1)
    num_classes = logits.shape[1]
    one_hot = F.one_hot(target, num_classes) # [B, K]
    return torch.max(logits - one_hot * LARGE_CONST, dim=1) 
    
def get_target_value(logits, target): 
    """
    return value at the target index 
    """
    assert(len(logits.shape) == 2)
    assert(len(target.shape) == 1)
    num_classes = logits.shape[1]
    one_hot = F.one_hot(target, num_classes) # [B,K]
    reverse_one_hot = 1.0 - one_hot
    return torch.max(logits - reverse_one_hot * LARGE_CONST, dim=1)

def gen_mask_v2(logits, target, margin=0.05): 
    """
    Generate mask regarding current prediction probability, output shape: [batch_size,]
    Returns: 
        mask =   where (max_non_target_pred - pred_at_target - margin > 0 , 1.0, margin) 
        max_non_target_pred > pred_at_target + margin : incorrect prediction --> 1.0 
        max_non_target_pred < pred_at_target + margin: correct prediction --> margin 
    """
    assert(len(logits.shape) == 2)
    assert(len(target.shape) == 1)
    pred = torch.softmax(logits, dim=1)
    max_non, _ = max_non_target(pred, target)
    pred_true, _ =  get_target_value(pred, target)
    # return torch.relu(max_non - pred_true - margin) + margin
    return torch.where(max_non - pred_true - margin > 0, 1.0, margin)

def gradient_maps(models, inputs, targets):
    """
    Generate 2D maps 
    Args: 
        models: list of K models 
        inputs: input tennsor, shape [B, C, W, H]
        targets: label tensor, shape [B]
    Return: 
        2D tensor [K, K] where [i,j] = cosine(grad_i, grad_j)
        where grad_i = gradient(loss(model_i, input), input_i)
    """
    grads = []
    X = Variable(inputs.data.clone(), requires_grad=True).to(inputs.device)
    
    for model in models: 
        if X.grad is not None:
            X.grad.data.zero_()
        model.zero_grad()

        with torch.enable_grad():
            pred = model(X)
            loss =  F.cross_entropy(pred, targets, reduction='sum')
        grad = torch.autograd.grad(loss, X, retain_graph=True)[0]
        grad = torch.flatten(grad, start_dim=1) # [B, C*W*H]
        grads.append(grad.data.clone()) # 

    grads = torch.stack(grads, dim=1) # [B, K, C*W*H]
    grads = grads.detach()

    grads_A = torch.unsqueeze(grads, dim=1) # [B, 1, K, C*W*H]
    grads_B = torch.unsqueeze(grads, dim=2) # [B, K, 1, C*W*H]

    sim = torch.cosine_similarity(grads_B, grads_A, dim=-1) # [B, K, K]
    sim_mean = torch.mean(sim, dim=0) # [K, K]
    sim_std = torch.std(sim, dim=0) # [K, K]

    return sim_mean.detach().cpu().numpy(), sim_std.detach().cpu().numpy()

def my_cosine_similarity(grads_A, grads_B, epsilon=1e-8):
    assert(grads_A.shape[0] == grads_B.shape[0])
    assert(grads_A.shape[-1] == grads_B.shape[-1])
    assert(grads_A.shape[1] == grads_B.shape[2])
    assert(grads_A.shape[2] == grads_B.shape[1])

    norm_grads_A = torch.norm(grads_A, p=2, dim=-1)
    norm_grads_B = torch.norm(grads_B, p=2, dim=-1)

    norm_grads_A = torch.unsqueeze(norm_grads_A, dim=3)
    norm_grads_B = torch.unsqueeze(norm_grads_B, dim=3)

    output = torch.divide(grads_A, norm_grads_A+epsilon) * torch.divide(grads_B, norm_grads_B+epsilon)
    output = torch.sum(output, dim=-1)

    return output

def test_cosine_similarity():
    grads_A = torch.randn(size=(4,1,3,500))
    # grads_B = torch.randn(size=(4,3,1,500))
    grads_B = torch.transpose(grads_A, 1, 2)

    sim1 = torch.cosine_similarity(grads_B, grads_A, dim=-1) 
    sim2 = my_cosine_similarity(grads_B, grads_A) 

    # print(sim1-sim2)
    print(torch.mean(sim1, dim=0))
    print('----')
    print(torch.mean(sim2, dim=0))

def test_2():
    grads = []
    for _ in range(3): 
        grad = torch.randn(size=(4,500))
        grads.append(grad)
    
    grads = torch.stack(grads, dim=1) # [B, K, C*W*H]

    grads_A = torch.unsqueeze(grads, dim=1) # [B, 1, K, C*W*H]
    grads_B = torch.unsqueeze(grads, dim=2) # [B, K, 1, C*W*H]    

    print('grads', grads.shape)
    print('grads_A', grads_A.shape)
    print('grads_B', grads_B.shape)

    sim1 = torch.cosine_similarity(grads_B, grads_A, dim=-1) 
    sim2 = my_cosine_similarity(grads_B, grads_A) 

    # print(sim1-sim2)
    print(torch.mean(sim1, dim=0))
    print(torch.std(sim1, dim=0))

    grad_map_adv_m = torch.mean(sim1, dim=0)
    grad_map_adv_std = torch.std(sim1, dim=0)

    grad_map_adv_m = np.asarray([[1,2,3],[4,5,6],[7,8,9]])
    print(grad_map_adv_m)
    
    import matplotlib.pyplot as plt 
    import matplotlib

    fig, ax = plt.subplots(figsize=(8,8))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list('my_colormap',
                                            ['blue','black','red'],
                                            256)

    img = plt.imshow(grad_map_adv_m,interpolation='nearest',
                    cmap = cmap,
                    origin='lower')
    plt.colorbar(img, cmap=cmap)
    for r in range(grad_map_adv_m.shape[0]): 
        for c in range(grad_map_adv_m.shape[1]): 
            ax.text(grad_map_adv_m.shape[0]-c-1,grad_map_adv_m.shape[1]-r-1,'{:.2f}+/-{:.2f}'.format(grad_map_adv_m[r,c], grad_map_adv_std[r,c]), color='white')
    plt.savefig('test.png', dpi=300)
    plt.close()

if __name__ == "__main__":
    test_2()
