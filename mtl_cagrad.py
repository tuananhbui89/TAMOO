import copy 
import random
import numpy as np 
import torch 
import torch.nn.functional as F 
from torch.autograd import Variable
from scipy.optimize import minimize

from utils_cm import accuracy, member_accuracy
from utils_alg import max_non_target, get_target_value, random_select_target
from utils_alg import wrap_loss_fn
from utils_data import change_factor
from utils_proj import normalize_grad, proj_box, proj_box_appro, proj_prob_simplex

"""
    Generate adversarial examples with Multitask Learning method: CAGrad. 
    Ref: https://github.com/Cranial-XIX/CAGrad

"""

__all__ = [
    "cagrad_ens",  # ensemble attack over multiple models
]

def CAGradEns(models, X, y, device, attack_params): 
    return cagrad_ens(models, X, y, 
                    norm=attack_params['norm'], 
                    eps=attack_params['epsilon'], 
                    num_steps=attack_params['num_steps'], 
                    step_size=attack_params['step_size'],
                    loss_type=attack_params['loss_type'], 
                    beta= 1./attack_params['moo_lr'],
                    normalize=True, 
                    appro_proj=False, 
                    soft_label=False, 
                    task_oriented=False, 
                    m1_margin=attack_params['m1'], 
                    m2_margin=attack_params['m2'],
                    targeted=attack_params['targeted'], 
                    num_classes=attack_params['num_classes'],
                    )

def cagrad_ens(
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
    task_oriented=True, 
    m2_margin=10.0, 
    m1_margin=10.0,
    targeted=False, 
    num_classes=10,
):
    """
    CAGrad to generate adversarial examples of ensemble of models.
    
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

    eps = np.abs(eps)
    K = len(models)
    batch_size = x.shape[0]

    if rand_init:
        delta = eps * torch.rand_like(x, device=x.device, requires_grad=False) # random uniform 
    else:
        delta = torch.zeros_like(x, device=x.device, requires_grad=False)

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

    def _outer_max(delta, alpha=0.5, rescale=1):
        """
        Todo: Finding new perturbation that maximize the prediction loss 
            Can be considered as an one step of PGD 
        Args: 
            delta: previous perturbation 
        Note: 
            Need stop gradient of the output 
        """

        # print("outer min...")
        assert(delta.requires_grad is False)

        for _model in models: 
            _model.zero_grad()
        delta = Variable(delta.data, requires_grad=True).to(x.device)
        
        F = _update_F(delta)
        L = torch.sum(F, dim=0) # [K]
        
        # Collecting gradient w.r.t. each different/individual loss
        # Will not scaling with W 
         
        Gall = []
        for i in range(K): 
            models[i].zero_grad()
            if delta.grad is not None: 
                delta.grad.data.zero_()
            Gi = torch.autograd.grad(L[i], delta, retain_graph=True)[0] # shape of delta [B,C,W,H]
            Gi = Gi.flatten(start_dim=1) # [B, CWH]
            Gi = Gi.detach()

            Gall.append(Gi)


        """"
        CAGrad Algorithm
        Description of the original algorithm in Multi-task learning 
            - Step 1: Collecting grads w.r.t. to each task's loss. 
            Grad is on the shared params (encoder) only. 
            - Step 2: Store list of grads to a place holder 
            - Step 3: clean grad in the shared params 
            - Step 4: (Main job) Calculating cagrad 
            - Step 5: Overwrite_grad 
        We will adapt this algorithm to AML context. 
        Ref: https://github.com/Cranial-XIX/CAGrad/blob/main/nyuv2/utils.py

        """
        def cagrad(grads, n_tasks, alpha=0.5, rescale=1):
            assert(len(grads.shape) == 2) # [CWH, K]
            assert(grads.shape[1] == n_tasks)

            GG = grads.t().mm(grads).cpu()  # [num_tasks, num_tasks]
            g0_norm = (GG.mean() + 1e-8).sqrt()  # norm of the average gradient

            x_start = np.ones(n_tasks) / n_tasks
            bnds = tuple((0, 1) for x in x_start)
            cons = {"type": "eq", "fun": lambda x: 1 - sum(x)}
            A = GG.numpy()
            b = x_start.copy()
            c = (alpha * g0_norm + 1e-8).item()

            def objfn(x):
                return (
                    x.reshape(1, n_tasks).dot(A).dot(b.reshape(n_tasks, 1))
                    + c
                    * np.sqrt(
                        x.reshape(1, n_tasks).dot(A).dot(x.reshape(n_tasks, 1))
                        + 1e-8
                    )
                ).sum()

            res = minimize(objfn, x_start, bounds=bnds, constraints=cons)
            w_cpu = res.x
            ww = torch.Tensor(w_cpu).to(grads.device)
            gw = (grads * ww.view(1, -1)).sum(1)
            gw_norm = gw.norm()
            lmbda = c / (gw_norm + 1e-8)
            g = grads.mean(1) + lmbda * gw
            if rescale == 0:
                return g
            elif rescale == 1:
                return g / (1 + alpha ** 2)
            else:
                return g / (1 + alpha)

        Gall = torch.stack(Gall, dim=2) # [B, CWH, K]

        merged_grad = []
        for im in range(batch_size):
            g_im = cagrad(grads=Gall[im], n_tasks=K, alpha=alpha, rescale=rescale)
            merged_grad.append(g_im)
        
        merged_grad = torch.stack(merged_grad, dim=0)
        assert len(merged_grad.shape) == 2
        assert merged_grad.shape[0] == batch_size
        assert merged_grad.shape[1] == Gall.shape[1]

        norm_grad_common = torch.norm(merged_grad, p=2, dim=-1) # [B]
        norm_grad_common = torch.mean(norm_grad_common, dim=0)

        merged_grad = torch.reshape(merged_grad, shape=delta.shape)        

        # normalize the gradients
        if normalize:
            merged_grad = normalize_grad(merged_grad, norm)

        delta = delta + step_size * merged_grad # Ascent the loss  
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

        return delta, norm_grad_common

    """
    Todo: Main loop, at each step
        Call outer_min to update delta 
    """

    for _ in range(num_steps): 
        assert(delta.requires_grad is False)
        delta, norm_grad_common = _outer_max(delta)
        assert(delta.requires_grad is False)

        for im in range(len(models)): 
            pred = models[im](x+delta) 
            adv_logits = models[im](x+delta)
            nat_logits = models[im](x)
            loss = wrap_loss_fn(y, adv_logits, nat_logits, reduction='sum', loss_type=loss_type)
            sar = 100. - accuracy(adv_logits, y)[0]
            log['model{}:loss'.format(im)].append(loss.item())
            log['model{}:sar'.format(im)].append(sar.item())

        all_correct, all_incorrect, acc_avg, _ = member_accuracy(models, x+delta, y)
        log['sar_all'].append(100.*all_incorrect.item())
        log['sar_atleastone'].append(100. - 100.*all_correct.item())
        log['sar_avg'].append(100. - 100. * acc_avg.item())
        log['norm_grad_common'].append(norm_grad_common.item())
    
    return x + delta, log
