import numpy as np 
import torch 
import torch.nn.functional as F 
from torch.autograd import Variable

from utils_cm import accuracy, member_accuracy
from utils_alg import max_non_target, get_target_value, random_select_target
from utils_alg import wrap_loss_fn
from utils_data import change_factor
from utils_proj import normalize_grad, proj_box, proj_box_appro, proj_prob_simplex


__all__ = [
    "moo_ens",  # ensemble attack over multiple models
    "moo_uni",  # universal perturbation over multiple images
    "moo_eot",  # robust attack over multiple data transformation
]

def MOOEns(models, X, y, device, attack_params): 
    return moo_ens(models, X, y, 
                    norm=attack_params['norm'], 
                    eps=attack_params['epsilon'], 
                    num_steps=attack_params['num_steps'], 
                    step_size=attack_params['step_size'],
                    loss_type=attack_params['loss_type'], 
                    beta= 1./attack_params['moo_lr'],
                    initial_w=attack_params['initial_w'],
                    normalize=True, 
                    appro_proj=False, 
                    soft_label=False, 
                    task_oriented=False, 
                    m1_margin=attack_params['m1'], 
                    targeted=attack_params['targeted'], 
                    num_classes=attack_params['num_classes'],
                    )

def MOOTOEns(models, X, y, device, attack_params): 
    return moo_ens(models, X, y, norm=attack_params['norm'], 
                    eps=attack_params['epsilon'], 
                    num_steps=attack_params['num_steps'], 
                    step_size=attack_params['step_size'],
                    loss_type=attack_params['loss_type'], 
                    beta= 1./attack_params['moo_lr'],
                    initial_w=attack_params['initial_w'],
                    normalize=True, 
                    appro_proj=False, 
                    soft_label=False, 
                    task_oriented=True,  # CHANGE HERE 
                    m1_margin=attack_params['m1'], 
                    targeted=attack_params['targeted'], 
                    num_classes=attack_params['num_classes'],
                    )


def sm(W): 
    return torch.softmax(W, dim=1)

def moo_ens(
    models,
    x,
    y,
    norm=np.inf,
    eps=0.031,
    num_steps=20,
    step_size=0.007,
    beta=40,
    gamma=20,
    clip_min=0.0,
    clip_max=1.0,
    loss_type="ce",
    fixed_w=False,
    initial_w=None,
    appro_proj=False,
    normalize=False,
    rand_init=True,
    soft_label=True,
    task_oriented=True, 
    m1_margin=10.0,
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

    :param task_oriented: Using task oriented or not 
    :return: A tuple of tensors, contains adversarial samples for each input, 
             and converged domain weights.
    """
    if norm not in [np.inf, int(0), int(1), int(2)]:
        raise ValueError("Norm order must be either `np.inf`, 0, 1, or 2.")

    log = dict()
    log['sar_all'] = []
    log['sar_atleastone'] = []
    log['sar_avg'] = []
    log['norm_grad_common'] = []


    for im in range(len(models)): 
        log['model{}:loss'.format(im)] = []
        log['model{}:sar'.format(im)] = []
        log['model{}:w'.format(im)] = []
        log['model{}:w_std'.format(im)] = []
        log['model{}:norm'.format(im)] = []


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
        initial_w = torch.unsqueeze(initial_w, dim=0)
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

    def _update_F(delta):
        """
        Given K models and perturbation delta, getting loss of each model. 
        Using soft-label (predicted label) as a target 
        Using one-hot encoding inside the wrap_loss_fn --> input target is indice 
        Return: 
            stack losses in the dim=1, --> return a tensor [B,K] 
        Note: 
            using wrap_loss_fn with loss_type='ce', 'cw'
        """
        f = []
        for i, model in enumerate(models):
            nat_logits = model(x)
            adv_logits = model(x+delta)
            loss = wrap_loss_fn(targets[i], adv_logits, nat_logits, reduction='none', 
                                loss_type=loss_type) # [B, ]
            # Attention, change direction if targeted 
            loss = loss * targeted_mul
            f.append(loss)

        return torch.stack(f, dim=1) # [B, K]

    def _outer_max(delta, W):
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
        assert(W.requires_grad is False)
        for _model in models: 
            _model.zero_grad()
        delta = Variable(delta.data, requires_grad=True).to(x.device)
        
        F = _update_F(delta)
        loss_weighted = torch.sum(torch.multiply(sm(W), F), dim=1)
        loss_weighted = torch.sum(loss_weighted, dim=0)
        grad = torch.autograd.grad(loss_weighted, delta)[0] 
        # normalize the gradients
        if normalize:
            grad = normalize_grad(grad, norm)

        delta = delta + step_size * grad # Ascent the loss  
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

    def check_oscillation(x, j, k, y5, k3=0.75):
        t = torch.zeros(x.shape[1]).to(x.device)
        for counter5 in range(k):
          t += (x[j - counter5] > x[j - counter5 - 1]).float()

        return (t <= k * k3 * torch.ones_like(t)).float()

    def _outer_max_adaptive(delta, W, cur_step, meta=None):
        """
        Todo: Finding new perturbation that maximize the prediction loss with adaptive step size
            Requires additional paramters: 
                - current attack step 
                - total attack steps
                - list of previous losses 
                - current loss (to be calculated) 
                - first grad 
            To be easier to implement, we support the norm linf only. It means that 
                - alpha = 2 as in the original implementation  
        Note: 
            Need stop gradient of the output 
        """

        # If the current attack step is 0 (the first attack step), then set the step size to double the epsilon 
        alpha = 2 
        thr_decr = 0.75 
        size_decr = max(int(0.03 * num_steps), 1)
        n_iter_min = max(int(0.06 * num_steps), 1)
        n_iter_2 = max(int(0.22 * num_steps), 1)

        assert(delta.requires_grad is False)
        assert(W.requires_grad is False)
        if cur_step == 0:
            # Init global variables in the first attack step 
            assert(meta is None)
            meta = {}
            meta['step_size'] = alpha * eps * torch.ones([x.shape[0], 1, 1, 1]).to(x.device).detach()
            meta['x_best_adv'] = x + delta 
            meta['x_best_adv'] = meta['x_best_adv'].clone()
            meta['x_best'] = meta['x_best_adv'].clone()
            meta['x_adv_old'] = meta['x_best_adv'].clone()
            meta['loss_steps'] = torch.zeros([num_steps, x.shape[0]]).to(x.device)
            meta['loss_best_steps'] = torch.zeros([num_steps+1, x.shape[0]]).to(x.device)
            acc_steps = torch.zeros_like(meta['loss_best_steps'])
            meta['counter3'] = 0 

            # Calculate the current loss and gradient at the first step 
            for _model in models: 
                _model.zero_grad()
            delta = Variable(delta.data, requires_grad=True).to(x.device)
            
            F = _update_F(delta)
            meta['loss_indiv'] = torch.sum(torch.multiply(sm(W), F), dim=1)
            meta['loss_weighted'] = torch.sum(meta['loss_indiv'], dim=0)
            meta['prev_loss'] = torch.zeros_like(meta['loss_indiv'])

            meta['grad'] = torch.autograd.grad(meta['loss_weighted'], delta)[0] 
            # normalize the gradients
            if normalize:
                meta['grad'] = normalize_grad(meta['grad'], norm)

            meta['loss_best'] = meta['loss_indiv'].detach().clone()
            meta['loss_best_last_check'] = meta['loss_indiv'].detach().clone()
            meta['reduced_last_check'] = torch.ones_like(meta['loss_best_last_check'])
            meta['grad_best'] = meta['grad'].clone()
            n_reduced = 0
            meta['k'] = n_iter_2 + 0 


        # Update x_adv  
        x_adv = x + delta 
        x_adv = x_adv.detach()
        grad2 = x_adv - meta['x_adv_old'] 
        meta['x_adv_old'] = x_adv.clone() 

        a = 0.75 if cur_step > 0 else 1.0 

        x_adv_1 = x_adv + meta['step_size'] * torch.sign(meta['grad'])
        x_adv_1 = torch.clamp(torch.min(torch.max(x_adv_1,
                        x - eps), x + eps), clip_min, clip_max)
        
        x_adv_1 = torch.clamp(torch.min(torch.max(
            x_adv + (x_adv_1 - x_adv) * a + grad2 * (1 - a),
            x - eps), x + eps), 0.0, 1.0)
        
        x_adv = x_adv_1 + 0. 

        # Get second gradient? In the original implementation, it is used inside the for loop, while the previous grad was calculated outside the loop. 
        # grad is also be returned from the function, so we can use it outside the loop. 
        # update delta 
        delta = x_adv - x 
        delta = delta.detach() 
        delta = Variable(delta.data, requires_grad=True).to(x.device)

        F = _update_F(delta)
        meta['loss_indiv'] = torch.sum(torch.multiply(sm(W), F), dim=1)
        meta['loss_weighted'] = torch.sum(meta['loss_indiv'], dim=0)
        meta['grad'] = torch.autograd.grad(meta['loss_weighted'], delta)[0] 
        # normalize the gradients
        if normalize:
            meta['grad'] = normalize_grad(meta['grad'], norm)        

        # update meta['x_best_adv'], how to define the best adv in the case of attacking multiple models? 
        # In the original implementation, it choose the current adversarial example that is predicted incorrectly. 
        # In this implementation, we choose the adversarial example that has the highest weighted loss? 

        higher_loss = torch.where(meta['loss_indiv'] > meta['prev_loss'], 1.0, 0.0)
        ind_pred = higher_loss.nonzero().squeeze() 
        meta['x_best_adv'][ind_pred] = x_adv[ind_pred] + 0. 
        meta['prev_loss'][ind_pred] = meta['loss_indiv'][ind_pred] + 0.

        
        # Check step size 
        with torch.no_grad(): 
            y1 = meta['loss_indiv'].detach().clone()
            meta['loss_steps'][cur_step] = y1 + 0. 
            ind = (y1 > meta['loss_best']).nonzero().squeeze()
            meta['x_best'][ind] = x_adv[ind].clone()
            meta['grad_best'][ind] = meta['grad'][ind].clone()
            meta['loss_best'][ind] = y1[ind] + 0. 
            meta['loss_best_steps'][cur_step+1] = meta['loss_best'] + 0.

            meta['counter3'] += 1 

            if meta['counter3'] == meta['k']: 
                fl_oscillation = check_oscillation(meta['loss_steps'], cur_step, meta['k'],
                        meta['loss_best'], k3=thr_decr)
                fl_reduce_no_impr = (1. - meta['reduced_last_check']) * (
                    meta['loss_best_last_check'] >= meta['loss_best']).float()
                fl_oscillation = torch.max(fl_oscillation,
                    fl_reduce_no_impr)
                meta['reduced_last_check'] = fl_oscillation.clone()
                meta['loss_best_last_check'] = meta['loss_best'].clone()

                if fl_oscillation.sum() > 0:
                    ind_fl_osc = (fl_oscillation > 0).nonzero().squeeze()
                    meta['step_size'][ind_fl_osc] /= 2.0
                    n_reduced = fl_oscillation.sum()

                    x_adv[ind_fl_osc] = meta['x_best'][ind_fl_osc].clone()
                    meta['grad'][ind_fl_osc] = meta['grad_best'][ind_fl_osc].clone()

                meta['k'] = max(meta['k'] - size_decr, n_iter_min)

                meta['counter3'] = 0 

        delta = x_adv - x 
        delta = delta.detach()
        return delta, meta 
    
    def _get_mask(delta):
        mask = []
        for i, model in enumerate(models):
            adv_logits = model(x+delta)
            pred = torch.argmax(adv_logits, dim=1)
            _mask = torch.where(pred == targets[i], 0.0, 1.0)
            mask.append(_mask)
        return torch.stack(mask, dim=1) # [B, K]

    def _inner_min(delta, W, gamma, beta):
        """
        Todo: MOO Solver, using Gradient Descent 
        Find W to minimize the gradient norm 

        """
        if fixed_w:
            # average case or static heuristic weights
            return W  
        # print("inner max...")
        assert(delta.requires_grad is False)
        for model in models:
            model.zero_grad()
        delta.requires_grad_(True)

        assert(W.requires_grad is False)
        W.requires_grad_(True)

        F = _update_F(delta) # [B, K]
        L = torch.sum(F, dim=0) # [K]
        
        Gall = []
        for i in range(K): 
            models[i].zero_grad()
            if delta.grad is not None: 
                delta.grad.data.zero_()
            Gi = torch.autograd.grad(L[i], delta, retain_graph=True)[0] # shape of delta [B,C,W,H]
            Gi = Gi.flatten(start_dim=1) # [B, CWH]
            Gi = Gi.detach()

            Gall.append(Gi)

        Gall = torch.stack(Gall, dim=1) # [B, K, CWH]
        GNorm = torch.unsqueeze(sm(W), dim=2) * Gall # [B,K,1] * [B,K,CWH]
        GNorm = torch.sum(GNorm, dim=1) # [B,CWH]
        GNorm = torch.sum(torch.square(GNorm)) # [1]
        GNorm += gamma * torch.sum(torch.square(sm(W) - 1/K)) # Optional, uniform regularization 
        gW = torch.autograd.grad(GNorm, W)[0] # [B, K]
        W = W - 1.0 / beta * gW # Gradient descent, update W 

        # Measuring the norm of gradient 
        norm_grad = torch.norm(Gall, p=2, dim=-1) # [B, K]
        norm_grad = torch.mean(norm_grad, dim=0) # [K]
        norm_grad_common = torch.unsqueeze(sm(W), dim=2) * Gall #  [B,K,1] * [B,K,CWH]
        norm_grad_common = torch.sum(norm_grad_common, dim=1) # [B, CWH]
        norm_grad_common = torch.norm(norm_grad_common, p=2, dim=-1) # [B]
        norm_grad_common = torch.mean(norm_grad_common, dim=0) # [1]

        if delta.grad is not None: 
            delta.grad.data.zero_()        
        delta.requires_grad_(False)
        W = W.detach()
        return W, norm_grad, norm_grad_common

    def _inner_min_to(delta, W, gamma, beta):
        """
        Todo: MOO Solver, using Gradient Descent with task oriented 
        Find W to minimize the gradient norm 

        """
        if fixed_w:
            # average case or static heuristic weights
            return W  
        # print("inner max...")
        assert(delta.requires_grad is False)
        for model in models:
            model.zero_grad()
        delta.requires_grad_(True)

        assert(W.requires_grad is False)
        W.requires_grad_(True)
        F = _update_F(delta) # [B, K]
        L = torch.sum(F, dim=0) # [K]

        mask = _get_mask(delta)
        mask = mask.detach()

        Gall = []
        for i in range(K): 
            models[i].zero_grad()
            if delta.grad is not None: 
                delta.grad.data.zero_()
            Gi = torch.autograd.grad(L[i], delta, retain_graph=True)[0] # shape of delta [B,C,W,H]
            Gi = Gi.flatten(start_dim=1) # [B, CWH]
            Gi = Gi.detach()

            Gall.append(Gi)

        Gall = torch.stack(Gall, dim=1) # [B, K, CWH]

        for _ in range(10): 
            # GNorm = (1- torch.unsqueeze(mask, dim=2)) * torch.unsqueeze(sm(W), dim=2) * Gall # [B,K,1] * [B,K,1] * [B,K,CWH]
            GNorm = torch.unsqueeze(sm(W), dim=2) * Gall # [B,K,1] * [B,K,1] * [B,K,CWH]
            GNorm = torch.sum(GNorm, dim=1) # [B,CWH]
            GNorm = torch.sum(torch.square(GNorm)) # [1]
            
            smW = sm(W)
            # # Minimize weights w.r.t. successful tasks (i.e., masks==1)
            GNorm += m1_margin * torch.sum(mask * torch.square(smW))
            GNorm += m1_margin * torch.sum(torch.square(1 - torch.sum((1-mask) * smW, dim=1)) / (torch.sum(1 - mask, dim=1)+1e-8))

            ## Uniform reg for unsuccessful tasks --> maximize entropy
            entropy = - smW * torch.log(smW) # [B,K]
            reg = torch.sum(torch.sum((1-mask) * entropy, dim=1), dim=0)
            GNorm -= gamma * reg # Optional, uniform regularization 

            if W.grad is not None:
                W.grad.data.zero_()

            gW = torch.autograd.grad(GNorm, W, retain_graph=True)[0] # [B, K]
            W = W - 1.0 / beta * gW # Gradient descent, update W 

        # Measuring the norm of gradient 
        norm_grad = torch.norm(Gall, p=2, dim=-1) # [B, K]
        norm_grad = torch.mean(norm_grad, dim=0) # [K]
        norm_grad_common = torch.unsqueeze(sm(W), dim=2) * Gall #  [B,K,1] * [B,K,CWH]
        norm_grad_common = torch.sum(norm_grad_common, dim=1) # [B, CWH]
        norm_grad_common = torch.norm(norm_grad_common, p=2, dim=-1) # [B]
        norm_grad_common = torch.mean(norm_grad_common, dim=0) # [1]

        if delta.grad is not None: 
            delta.grad.data.zero_()        
        delta.requires_grad_(False)
        W = W.detach()
        return W, norm_grad, norm_grad_common

    """
    Todo: Main loop, at each step
        Call outer_min to update delta 
        Call inner_max to update W 
    """
    meta = None

    for cur_step in range(num_steps): 
        assert(delta.requires_grad is False)
        delta, meta = _outer_max_adaptive(delta, W, cur_step, meta)
        assert(delta.requires_grad is False)
        if task_oriented:
            W, norm_grad, norm_grad_common = _inner_min_to(delta, W, gamma, beta)
        else: 
            W, norm_grad, norm_grad_common = _inner_min(delta, W, gamma, beta)
        
        assert(len(norm_grad) == len(models))

        for im in range(len(models)): 
            pred = models[im](x+delta) 
            adv_logits = models[im](x+delta)
            nat_logits = models[im](x)
            loss = wrap_loss_fn(y, adv_logits, nat_logits, reduction='sum', loss_type=loss_type)
            sar = 100. - accuracy(adv_logits, y)[0]
            log['model{}:loss'.format(im)].append(loss.item())
            log['model{}:sar'.format(im)].append(sar.item())
            log['model{}:w'.format(im)].append(torch.mean(sm(W)[:,im]).item())
            log['model{}:w_std'.format(im)].append(torch.std(sm(W)[:,im]).item())
            log['model{}:norm'.format(im)].append(norm_grad[im].item())

        all_correct, all_incorrect, acc_avg, _ = member_accuracy(models, x+delta, y)
        log['sar_all'].append(100.*all_incorrect.item())
        log['sar_atleastone'].append(100. - 100.*all_correct.item())
        log['sar_avg'].append(100. - 100. * acc_avg.item())
        log['norm_grad_common'].append(norm_grad_common.item())

    return x + delta, log
