from utils_alg import wrap_loss_fn
import numpy as np 
import torch 
import torch.nn.functional as F 
from utils_cm import accuracy, member_accuracy
from torch.autograd import Variable
from utils_data import change_factor
from utils_alg import random_select_target
from utils_proj import normalize_grad, proj_box, proj_box_appro, proj_prob_simplex

__all__ = [
    "minmax_ens",  # ensemble attack over multiple models
    "minmax_uni",  # universal perturbation over multiple images
    "minmax_eot",  # robust attack over multiple data transformation
]

def MinMaxEns(models, X, y, device, attack_params): 
    return minmax_ens(models, X, y, norm=attack_params['norm'], 
                    eps=attack_params['epsilon'], 
                    num_steps=attack_params['num_steps'], 
                    step_size=attack_params['step_size'],
                    loss_type=attack_params['loss_type'], 
                    beta= 1./attack_params['moo_lr'],
                    normalize=True, 
                    appro_proj=False, 
                    soft_label=False, 
                    targeted=attack_params['targeted'], 
                    num_classes=attack_params['num_classes'])

def MinMaxUni(model, X, y, device, attack_params): 
    return minmax_uni(model, X, y, norm=attack_params['norm'], 
                    eps=attack_params['epsilon'], 
                    num_steps=attack_params['num_steps'], 
                    step_size=attack_params['step_size'],
                    loss_type=attack_params['loss_type'], 
                    beta= 1./attack_params['moo_lr'],
                    normalize=True, 
                    appro_proj=False, 
                    soft_label=False,
                    targeted=attack_params['targeted'], 
                    num_classes=attack_params['num_classes'])

def MinMaxEoT(model, X, y, Trf, device, attack_params): 
    return minmax_eot(model, X, y, Trf, norm=attack_params['norm'], 
                    eps=attack_params['epsilon'], 
                    num_steps=attack_params['num_steps'], 
                    step_size=attack_params['step_size'],
                    loss_type=attack_params['loss_type'], 
                    beta= 1./attack_params['moo_lr'],
                    normalize=True, 
                    appro_proj=False, 
                    soft_label=False, 
                    eotsto=attack_params['eotsto'],
                    targeted=attack_params['targeted'], 
                    num_classes=attack_params['num_classes'])

def minmax_ens(
    models,
    x,
    y,
    norm=np.inf,
    eps=0.031,
    num_steps=20,
    step_size=0.007,
    beta=40,
    gamma=3,
    clip_min=0.0,
    clip_max=1.0,
    loss_type="ce",
    fixed_w=False,
    initial_w=None,
    appro_proj=False,
    normalize=False,
    rand_init=True,
    soft_label=True,
    targeted=False,
    num_classes=10,
):
    """
    Min-max ensemble attack over via APGDA.
    :param models: A list of victim models that return the logits.
    :param x: The input placeholder.
    :param norm: The L_p norm: np.inf, 2, 1, 0.  
    :param eps: The scale factor for noise.
    :param num_steps: Number of steps to run attack generation.
    :param step_size: step size for outer minimization (update perturation).
    :param beta: 1/beta is step size for inner maximization (update domain weights).
    :param gamma: regularization coefficient to balance avg-case and worst-case.
    :param clip_min: The minimum value in output.
    :param clip_max: The maximum value in output.
    :param loss_type: Adversarial loss function: "ce" or "cw".
    :param models_suffix: The suffixes for victim models, e.g., "ABC" for three models.
    :param fixed_w: Fixing domain weights and does not update.
    :param initial_w: The initial domain weight vector.
    :param appro_proj: Use approximate solutions for box constraint (l_p) projection.
    :param normalize: Normalize the gradients when optimizing perturbations.
    :param rand_init: Random normal initialization for perturbations.
    :return: A tuple of tensors, contains adversarial samples for each input, 
             and converged domain weights.
    """
    if norm not in [np.inf, int(0), int(1), int(2)]:
        raise ValueError("Norm order must be either `np.inf`, 0, 1, or 2.")

    log = dict()
    log['sar_all'] = []
    log['sar_atleastone'] = []
    log['sar_avg'] = []

    for im in range(len(models)): 
        log['model{}:loss'.format(im)] = []
        log['model{}:sar'.format(im)] = []
        log['model{}:w'.format(im)] = []

    eps = np.abs(eps)
    K = len(models)
    if rand_init:
        delta = eps * torch.rand_like(x, device=x.device, requires_grad=False) # random uniform 
    else:
        delta = torch.zeros_like(x, device=x.device, requires_grad=False)

    batch_size = x.shape[0]

    if initial_w is None:
        W = torch.ones(size=(batch_size, K)).to(x.device) / K
    else:
        """
        repeat along first dimension, given a initial_w shape [K]
        """
        assert(len(initial_w.shape)==1)
        initial_w = torch.tensor(initial_w, device=x.device, requires_grad=False)
        initial_w = torch.unsqueeze(initial_w, dim=1)
        W = torch.repeat_interleave(initial_w, repeats=batch_size, dim=0)

    if soft_label:
        targets = []
        for i in range(K):
            """
                Given K models, get one_hot vector of predicted target (soft-label) of each model
                Using one-hot encoding inside the wrap_loss_fn --> target is indince 
                return as a list 
            """
            pred = models[i](x)
            target = torch.argmax(pred, dim=1)
            targets.append(target)
    else: 
        targets = [y for _ in range(K)]
    
    """ Add targeted attacks 
    if targeted:
        targetes is random from non-ground truth set 
        maximizing loss --> minimizing loss 
    """
    targeted_mul = -1 if targeted else 1 
    if targeted:
        target = random_select_target(targets[0], num_classes=num_classes)
        targets = [target for _ in range(K)]

    def _update_F(delta, W):
        """
        Given K models and perturbation delta, getting loss of each model. 
        Using soft-label (predicted label) as a target 
        Using one-hot encoding inside the wrap_loss_fn --> input target is indice 
        Return: 
            stack losses in the dim=1, --> return a tensor [B,K] 
        Note: 
            using negative of normal loss. e.g., minimize - cross_entropy --> maximize cross_entropy 
            using wrap_loss_fn with loss_type='ce', 'cw'
        """
        f = []
        for i, model in enumerate(models):
            nat_logits = model(x)
            adv_logits = model(x+delta)
            loss = - wrap_loss_fn(targets[i], adv_logits, nat_logits, reduction='none', 
                                loss_type=loss_type) # [B, ]
            # Attention, change direction if targeted attack 
            loss = loss * targeted_mul
            f.append(loss)

        return torch.stack(f, dim=1) # [B, K]

    def _outer_min(delta, W):
        """
        Todo: Finding new perturbation that maximize the prediction loss (differ what described in the 
        paper?)
            In the original implementation, it minimize the negative loss -->
            maximize the standard loss (?)
        Note: 
            Need stop gradient of the output 
        """

        # print("outer min...")
        assert(delta.requires_grad is False)
        for _model in models: 
            _model.zero_grad()
        delta = Variable(delta.data, requires_grad=True).to(x.device)
        
        F = _update_F(delta, W)
        loss_weighted = torch.sum(torch.multiply(W, F), dim=1)
        loss_weighted = torch.sum(loss_weighted, dim=0)
        grad = torch.autograd.grad(loss_weighted, delta)[0] 
        # normalize the gradients
        if normalize:
            grad = normalize_grad(grad, norm)

        delta = delta - step_size * grad # Descent but acctually Ascent 
        delta = delta.detach()
        
        # project perturbations
        if not appro_proj:
            # analytical solution
            delta = proj_box(delta, norm, eps, clip_min - x, clip_max - x)
        else:
            # approximate solution
            delta = proj_box_appro(delta, eps, norm)
            xadv = x + delta
            xadv = torch.clamp(xadv, clip_min, clip_max)
            delta = xadv - x

        return delta

    def _inner_max(delta, W, gamma, beta):
        """
        Todo: Update W that 
                maximize the attack loss (--> minimize standard loss)
                maximize the negative regularization (--> minimize the norm)
            Grad = F - gamma * (W - 1/K) # shape [B, K]
            W_{t+1} = W_{t} + 1/beta * Grad 
        """
        if fixed_w:
            # average case or static heuristic weights
            return W  
        # print("inner max...")
        F = _update_F(delta, W)
        F = F.detach()
        G = F - gamma * (W - 1 / K)
        W += 1.0 / beta * G
        W = proj_prob_simplex(W, batch_size, K)
        return W

    """
    Todo: Main loop, at each step
        Call outer_min to update delta 
        Call inner_max to update W 
    """

    for _ in range(num_steps): 
        delta = _outer_min(delta, W)
        W = _inner_max(delta, W, gamma, beta)
        for im in range(len(models)): 
            pred = models[im](x+delta) 
            adv_logits = models[im](x+delta)
            nat_logits = models[im](x)
            loss = wrap_loss_fn(y, adv_logits, nat_logits, reduction='sum', loss_type=loss_type)
            sar = 100. - accuracy(adv_logits, y)[0]
            log['model{}:loss'.format(im)].append(loss.item())
            log['model{}:sar'.format(im)].append(sar.item())
            log['model{}:w'.format(im)].append(torch.mean(W[:,im]).item())

        all_correct, all_incorrect, acc_avg, _ = member_accuracy(models, x+delta, y)
        log['sar_all'].append(100.*all_incorrect.item())
        log['sar_atleastone'].append(100. - 100.*all_correct.item())
        log['sar_avg'].append(100. - 100.*acc_avg.item())
    
    return x + delta, log

def minmax_uni(
    model,
    x,
    y,
    norm=np.inf,
    eps=0.031,
    num_steps=20,
    step_size=0.007,
    beta=40,
    gamma=3,
    clip_min=0.0,
    clip_max=1.0,
    loss_type="ce",
    fixed_w=False,
    initial_w=None,
    appro_proj=False,
    normalize=False,
    rand_init=True,
    soft_label=True,
    targeted=False, 
    num_classes=10,
):
    """
    Min-max universarial adversarial perturbations via APGDA.
    :param model: A single victim model that return the logits.
    :param x: The input, shape [batch_size, K, C, width, height].
    :param y: The target, shape [batch_size, K]
    :param norm: The L_p norm: np.inf, 2, 1, 0.  
    :param eps: The scale factor for noise.
    :param num_steps: Number of steps to run attack generation.
    :param step_size: step size for outer minimization (update perturation).
    :param beta: 1/beta is step size for inner maximization (update domain weights).
    :param gamma: regularization coefficient to balance avg-case and worst-case.
    :param clip_min: The minimum value in output.
    :param clip_max: The maximum value in output.
    :param loss_type: Adversarial loss function: "ce" or "cw".
    :param models_suffix: The suffixes for victim models, e.g., "ABC" for three models.
    :param fixed_w: Fixing domain weights and does not update.
    :param initial_w: The initial domain weight vector.
    :param appro_proj: Use approximate solutions for box constraint (l_p) projection.
    :param normalize: Normalize the gradients when optimizing perturbations.
    :param rand_init: Random normal initialization for perturbations.
    :return: A tuple of tensors, contains adversarial samples for each input, 
             and converged domain weights.
    """
    if norm not in [np.inf, int(0), int(1), int(2)]:
        raise ValueError("Norm order must be either `np.inf`, 0, 1, or 2.")

    log = dict()
    log['loss'] = []
    log['sar'] = []
    log['w_mean'] = []
    log['w_std'] = []
    log['max_sar'] = 0

    eps = np.abs(eps)
    batch_size, K, C, width, height = x.shape 

    if rand_init:
        delta = eps * torch.rand(size=[batch_size,C,width,height], device=x.device, requires_grad=False) # random uniform 
    else:
        delta = torch.zeros(size=[batch_size,C,width,height], device=x.device, requires_grad=False)

    if initial_w is None:
        W = torch.ones(size=(batch_size, K)).to(x.device) / K
    else:
        """
        repeat along first dimension, given a initial_w shape [K]
        """
        assert(len(initial_w.shape)==1)
        initial_w = torch.tensor(initial_w, device=x.device, requires_grad=False)
        initial_w = torch.unsqueeze(initial_w, dim=1)
        W = torch.repeat_interleave(initial_w, repeats=batch_size, dim=0)

    if soft_label:
        targets = []
        for i in range(K):
            """
                Given a single model and K sets of samples, get predicted target of each set 
                Using one-hot encoding inside the wrap_loss_fn --> target is indince 
                return as a list 
            """
            pred = model(x[:,i])
            target = torch.argmax(pred, dim=1)
            targets.append(target)
        targets = torch.stack(targets, dim=1)
        assert(len(targets.shape) == 2)
    else: 
        assert(len(y.shape) == 2)
        targets = y 
    
    """ Add targeted attacks 
    if targeted:
        targetes is random from non-ground truth set 
        maximizing loss --> minimizing loss 
    """
    targeted_mul = -1 if targeted else 1 
    if targeted:
        _targets = torch.reshape(targets, [batch_size * K,])
        target = random_select_target(_targets, num_classes=num_classes)
        targets = torch.reshape(targets, [batch_size, K])

    def _update_F(delta, W):
        """
        Given a single model and perturbation delta, getting loss of each set of samples 
        Using soft-label (predicted label) as a target 
        Using one-hot encoding inside the wrap_loss_fn --> input target is indice 
        Return: 
            stack losses in the dim=1, --> return a tensor [B,K] 
        Note: 
            using negative of normal loss. e.g., minimize - cross_entropy --> maximize cross_entropy 
            using wrap_loss_fn with loss_type='ce', 'cw'
        """
        f = []
        for i in range(K):
            nat_logits = model(x[:,i])
            adv_logits = model(x[:,i]+delta)
            loss = - wrap_loss_fn(targets[:,i], adv_logits, nat_logits, reduction='none', 
                                loss_type=loss_type) # [B, ]
            # Attention, change direction if targeted 
            loss = loss * targeted_mul
            f.append(loss)

        return torch.stack(f, dim=1) # [B, K]

    def _outer_min(delta, W):
        """
        Todo: Finding new perturbation that maximize the prediction loss (differ what described in the 
        paper?)
            In the original implementation, it minimize the negative loss -->
            maximize the standard loss (?)
        Note: 
            Need stop gradient of the output 
        """

        # print("outer min...")
        assert(delta.requires_grad is False)
        model.zero_grad()
        delta = Variable(delta.data, requires_grad=True).to(x.device)
        
        F = _update_F(delta, W)
        loss_weighted = torch.sum(torch.multiply(W, F), dim=1)
        loss_weighted = torch.sum(loss_weighted, dim=0)
        grad = torch.autograd.grad(loss_weighted, delta)[0] 
        # normalize the gradients
        if normalize:
            grad = normalize_grad(grad, norm)

        delta = delta - step_size * grad # Descent but acctually Ascent 
        delta = delta.detach()
        
        for i in range(K):
            # project perturbations
            if not appro_proj:
                # analytical solution
                delta = proj_box(delta, norm, eps, clip_min - x[:,i], clip_max - x[:,i])
            else:
                # approximate solution
                delta = proj_box_appro(delta, eps, norm)
                xadv = x[:,i] + delta
                xadv = torch.clamp(xadv, clip_min, clip_max)
                delta = xadv - x[:,i]

        return delta

    def _inner_max(delta, W, gamma, beta):
        """
        Todo: Update W that 
                maximize the attack loss (--> minimize standard loss)
                maximize the negative regularization (--> minimize the norm)
            Grad = F - gamma * (W - 1/K) # shape [B, K]
            W_{t+1} = W_{t} + 1/beta * Grad 
        """
        if fixed_w:
            # average case or static heuristic weights
            return W  
        # print("inner max...")
        F = _update_F(delta, W)
        F = F.detach()
        G = F - gamma * (W - 1 / K)
        W += 1.0 / beta * G
        W = proj_prob_simplex(W, batch_size, K)
        return W

    """
    Todo: Main loop, at each step
        Call outer_min to update delta 
        Call inner_max to update W 
    """

    for _ in range(num_steps): 
        delta = _outer_min(delta, W)
        W = _inner_max(delta, W, gamma, beta)
        temp_x = torch.reshape(x, [batch_size*K, C, width, height])
        temp_xadv = torch.reshape(x+torch.stack([delta]*K, dim=1), [batch_size*K, C, width, height])
        temp_y = torch.reshape(y, [batch_size*K])
        adv_logits = model(temp_xadv)
        nat_logits = model(temp_x)
        loss = wrap_loss_fn(temp_y, adv_logits, nat_logits, reduction='sum', loss_type=loss_type)
        sar = 100. - accuracy(adv_logits, temp_y)[0]
        log['loss'].append(loss.item())
        log['sar'].append(sar.item())
        log['w_mean'].append(torch.mean(torch.mean(W, dim=0), dim=0).item())
        log['w_std'].append(torch.std(torch.mean(W,dim=0), dim=0).item())
        if sar.item() > log['max_sar']: 
            log['max_sar'] = sar.item()
    return x + torch.stack([delta]*K, dim=1), log


def minmax_eot(
    model,
    x,
    y,
    Trf,
    norm=np.inf,
    eps=0.031,
    num_steps=20,
    step_size=0.007,
    beta=40,
    gamma=3,
    clip_min=0.0,
    clip_max=1.0,
    loss_type="ce",
    fixed_w=False,
    initial_w=None,
    appro_proj=False,
    normalize=False,
    rand_init=True,
    soft_label=True,
    eotsto=False,
    targeted=False, 
    num_classes=10,
):
    """
    Min-max ensemble attack over via APGDA.
    :param models: A list of victim models that return the logits.
    :param x: The input placeholder.
    :param Trf: List of transofrmation 
    :param norm: The L_p norm: np.inf, 2, 1, 0.  
    :param eps: The scale factor for noise.
    :param num_steps: Number of steps to run attack generation.
    :param step_size: step size for outer minimization (update perturation).
    :param beta: 1/beta is step size for inner maximization (update domain weights).
    :param gamma: regularization coefficient to balance avg-case and worst-case.
    :param clip_min: The minimum value in output.
    :param clip_max: The maximum value in output.
    :param loss_type: Adversarial loss function: "ce" or "cw".
    :param models_suffix: The suffixes for victim models, e.g., "ABC" for three models.
    :param fixed_w: Fixing domain weights and does not update.
    :param initial_w: The initial domain weight vector.
    :param appro_proj: Use approximate solutions for box constraint (l_p) projection.
    :param normalize: Normalize the gradients when optimizing perturbations.
    :param rand_init: Random normal initialization for perturbations.
    :return: A tuple of tensors, contains adversarial samples for each input, 
             and converged domain weights.
    """
    if norm not in [np.inf, int(0), int(1), int(2)]:
        raise ValueError("Norm order must be either `np.inf`, 0, 1, or 2.")

    eps = np.abs(eps)

    K = len(Trf)

    log = dict()
    log['sar_all'] = []
    log['sar_atleastone'] = []
    log['sar_avg'] = []

    for im in range(K): 
        log['T{}:loss'.format(im)] = []
        log['T{}:sar'.format(im)] = []
        log['T{}:w'.format(im)] = []
    
    if rand_init:
        delta = eps * torch.rand_like(x, device=x.device, requires_grad=False) # random uniform 
    else:
        delta = torch.zeros_like(x, device=x.device, requires_grad=False)

    batch_size = x.shape[0]

    if initial_w is None:
        W = torch.ones(size=(batch_size, K)).to(x.device) / K
    else:
        """
        repeat along first dimension, given a initial_w shape [K]
        """
        assert(len(initial_w.shape)==1)
        initial_w = torch.tensor(initial_w, device=x.device, requires_grad=False)
        initial_w = torch.unsqueeze(initial_w, dim=1)
        W = torch.repeat_interleave(initial_w, repeats=batch_size, dim=0)

    if soft_label:
        targets = []
        for i in range(K):
            """
                Given K models, get one_hot vector of predicted target (soft-label) of each model
                Using one-hot encoding inside the wrap_loss_fn --> target is indince 
                return as a list 
            """
            pred = model(Trf[i](x))
            target = torch.argmax(pred, dim=1)
            targets.append(target)
    else: 
        targets = [y for _ in range(K)]

    """ Add targeted attacks 
    if targeted:
        targetes is random from non-ground truth set 
        maximizing loss --> minimizing loss 
    """
    targeted_mul = -1 if targeted else 1 
    if targeted:
        target = random_select_target(targets[0], num_classes=num_classes)
        targets = [target for _ in range(K)]

    def _update_F(delta, W):
        """
        Given K models and perturbation delta, getting loss of each model. 
        Using soft-label (predicted label) as a target 
        Using one-hot encoding inside the wrap_loss_fn --> input target is indice 
        Return: 
            stack losses in the dim=1, --> return a tensor [B,K] 
        Note: 
            using negative of normal loss. e.g., minimize - cross_entropy --> maximize cross_entropy 
            using wrap_loss_fn with loss_type='ce', 'cw'
        """
        f = []
        for i, T in enumerate(Trf):
            nat_logits = model(T(x))
            adv_logits = model(T(x+delta))
            loss = - wrap_loss_fn(targets[i], adv_logits, nat_logits, reduction='none', 
                                loss_type=loss_type) # [B, ]
            # Attention, change direction if targeted 
            loss = loss * targeted_mul
            f.append(loss)

        return torch.stack(f, dim=1) # [B, K]

    def _outer_min(delta, W):
        """
        Todo: Finding new perturbation that maximize the prediction loss (differ what described in the 
        paper?)
            In the original implementation, it minimize the negative loss -->
            maximize the standard loss (?)
        Note: 
            Need stop gradient of the output 
        """

        # print("outer min...")
        assert(delta.requires_grad is False)
        model.zero_grad()
        delta = Variable(delta.data, requires_grad=True).to(x.device)
        
        F = _update_F(delta, W)
        loss_weighted = torch.sum(torch.multiply(W, F), dim=1)
        loss_weighted = torch.sum(loss_weighted, dim=0)
        grad = torch.autograd.grad(loss_weighted, delta)[0] 
        # normalize the gradients
        if normalize:
            grad = normalize_grad(grad, norm)

        delta = delta - step_size * grad # Descent but acctually Ascent 
        delta = delta.detach()
        
        # project perturbations
        if not appro_proj:
            # analytical solution
            delta = proj_box(delta, norm, eps, clip_min - x, clip_max - x)
        else:
            # approximate solution
            delta = proj_box_appro(delta, eps, norm)
            xadv = x + delta
            xadv = torch.clamp(xadv, clip_min, clip_max)
            delta = xadv - x

        return delta

    def _inner_max(delta, W, gamma, beta):
        """
        Todo: Update W that 
                maximize the attack loss (--> minimize standard loss)
                maximize the negative regularization (--> minimize the norm)
            Grad = F - gamma * (W - 1/K) # shape [B, K]
            W_{t+1} = W_{t} + 1/beta * Grad 
        """
        if fixed_w:
            # average case or static heuristic weights
            return W  
        # print("inner max...")
        F = _update_F(delta, W)
        F = F.detach()
        G = F - gamma * (W - 1 / K)
        W += 1.0 / beta * G
        W = proj_prob_simplex(W, batch_size, K)
        return W

    """
    Todo: Main loop, at each step
        Call outer_min to update delta 
        Call inner_max to update W 
    """

    for _ in range(num_steps): 
        change_factor(eotsto)
        delta = _outer_min(delta, W)
        W = _inner_max(delta, W, gamma, beta)
        all_res = []
        for im, T in enumerate(Trf): 
            adv_logits = model(T(x+delta))
            nat_logits = model(T(x))
            loss = wrap_loss_fn(y, adv_logits, nat_logits, reduction='sum', loss_type=loss_type)
            sar = 100. - accuracy(adv_logits, y)[0]
            log['T{}:loss'.format(im)].append(loss.item())
            log['T{}:sar'.format(im)].append(sar.item())
            log['T{}:w'.format(im)].append(torch.mean(W[:,im]).item())

            _pred = torch.argmax(adv_logits, dim=1)
            _res = _pred.eq(y)
            _res = torch.unsqueeze(_res, dim=1).float()
            all_res.append(_res)

        all_res = torch.cat(all_res, dim=1)
        all_correct = torch.mean(torch.where(torch.prod(all_res, dim=1)==1,1.,0.))
        all_incorrect = torch.mean(torch.where(torch.sum(all_res, dim=1)==0,1.,0.))
        log['sar_all'].append(100.*all_incorrect.item())
        log['sar_atleastone'].append(100. - 100.*all_correct.item())
    
    return x + delta, log

