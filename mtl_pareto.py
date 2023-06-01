import copy 
import random
import numpy as np 
import torch 
import torch.nn.functional as F 
from torch.autograd import Variable

from utils_cm import accuracy, member_accuracy
from utils_alg import max_non_target, get_target_value, random_select_target
from utils_alg import wrap_loss_fn
from utils_data import change_factor
from utils_proj import normalize_grad, proj_box, proj_box_appro, proj_prob_simplex

from min_norm_solvers import MinNormSolver
from gradient_solvers import grad_descent_solv_batch

"""
    Generate adversarial examples with MTL method from paper: Pareto Multi-Task Learning
    Ref:  https://github.com/Xi-L/ParetoMTL

"""

__all__ = [
    "pareto_ens",  # ensemble attack over multiple models
]

def circle_points(r, n):
    """
    generate evenly distributed unit preference vectors for two tasks
    """
    circles = []
    for r, n in zip(r, n):
        t = np.linspace(0, 0.5 * np.pi, n)
        x = r * np.cos(t)
        y = r * np.sin(t)
        circles.append(np.c_[x, y])
    return circles

def k_circle_points(r,n,k): 
    """
    r: radius 
    n: number of points 
    k: number of vectors/tasks 
    Returns: 
        array size (n, k) with positive numbers 

    """
    assert type(r) is list 
    assert len(r) == 1 
    assert type(n) is list 
    assert len(n) == 1 

    circles = []
    t = np.linspace(0, 0.5 * np.pi, n[0]) 
    deltas = np.linspace(0, 0.5 * np.pi, k, endpoint=False)

    for r, n in zip(r, n):
        for ik in range(k):
            x = r * np.sin(t+deltas[ik]) # Using sin to make sure all points are positive 
            circles.append(np.c_[x])
    circles = np.concatenate(circles, axis=1)

    # OPTIONAL: Normalizing across tasks (i.e., axis=1)
    # circles = circles / np.expand_dims(np.sum(circles, axis=1), axis=1)

    circles = [circles]
    return circles    

def test_circle_points(): 
    circles = circle_points(r=[1], n=[5])
    print(circles)
    print(circles[0].shape) # output: (5,2)

    circles = k_circle_points(r=[1], n=[5], k=4)
    print(circles)
    print(circles[0].shape) # output: (5,2)

def get_d_paretomtl_org(grads,value,weights,i):
    """ calculate the gradient direction for ParetoMTL """
    """
        grads: original [K, d], in this setting [B, K, d]    
        value: loss, original [K], in this setting [B, K]
        weights: reference vector, original [npref, K], in this setting [B, npref, K]
    """
    assert len(weights.shape) == 2

    # check active constraints
    current_weight = weights[i] # original shape [K], in this setting [B, K]
    rest_weights = weights # original shape [npref, K], in this setting [B, npref, K]
    w = rest_weights - current_weight # original shape [npref, K], in this setting [B, npref, K]
    

    gx =  torch.matmul(w,value/torch.norm(value)) # original shape [npref], in this setting [B, npref] = matmul([B, npref, K], [B,K,1])
    idx = gx >  0  # original shape [npref], in this setting [B, npref]



    # calculate the descent direction
    if torch.sum(idx) <= 0:
        sol, nd = MinNormSolver.find_min_norm_element([grads[t] for t in range(len(grads))]) # CHANGE HERE, [[grads[t]]] to [grads[t]]
        return sol


    temp = torch.matmul(w[idx],grads) # original shape [num_positive, d]

    vec = torch.cat((grads, temp)) # original shape [K + num_positive, d]

    """
    MinNormSolver find_min_norm_element args: 
        - list of vectors, each vector has shape [d]
    """

    sol, nd = MinNormSolver.find_min_norm_element([vec[t] for t in range(len(vec))]) # CHANGE HERE, [[vec[t]]] to [vec[t]]

    weight0 =  sol[0] + torch.sum(torch.stack([sol[j] * w[idx][j - 2 ,0] for j in torch.arange(2, 2 + torch.sum(idx))]))
    weight1 =  sol[1] + torch.sum(torch.stack([sol[j] * w[idx][j - 2 ,1] for j in torch.arange(2, 2 + torch.sum(idx))]))
    weight = torch.stack([weight0,weight1])
    
    return weight


def get_d_paretomtl_pointwise(grads,value,weights,i):
    """ calculate the gradient direction for ParetoMTL """
    """
        grads: original [K, d], in this setting [B, K, d]    
        value: loss, original [K], in this setting [B, K]
        weights: reference vector, original [npref, K], in this setting [B, npref, K]
    """
    assert len(weights.shape) == 3
    assert len(grads.shape) == 3



    """"
    # check active constraints. Equation 7 in the paper
        Gj(θt)=(uj − uk).T L(θt) ≤0, ∀j=1,...,K
        where 
            u_k: current preference weight
            u_j: other preference weight 
            L(): current loss 
        if torch.sum(idx) <= 0 --> satisfy the condition in Equation 7. Just solve as normal 
    """
    current_weight = weights[i] # original shape [K], in this setting [B, K]
    rest_weights = weights # original shape [npref, K], in this setting [B, npref, K]
    w = rest_weights - current_weight # original shape [npref, K], in this setting [B, npref, K]
    
    _value = torch.unsqueeze(value, dim=2)
    gx = torch.matmul(w,_value/torch.norm(_value)) # [B, npref, 1]
    gx = torch.squeeze(gx) # [B, npref]
    idx = gx >  0  # original shape [npref], in this setting [B, npref]


    """
    Code explaination: 
        The below code is for solving Equation 9, finding direction of all activated constraints (i.e., idx == True)
    Step 1: Filter out all activated constraints 
        Note that, in the initialization stage in the original implementation, the author use 
        2 epochs to find the inital solution. 
    Step 2: In the second stage, they consider both: (i) all constraints (grads) and (ii) activated constraints (temp) 
    Step 3: Update the weight for current preference index. 
    """

    # temp = torch.matmul(w[idx],grads) # original shape [num_positive, d], activated constraints, 0 <= num_positive <= npref
    # [npref, K] x [K, d] = [npref, d]
    # print('temp', temp.shape)


    # vec = torch.cat((grads, temp)) # original shape [K + num_positive, d], all + activated constraints 
    # print('vec', vec.shape)


    """
    Change in the code compared to original implementation in order to adapt with AML setting. 
    - Using gradient descent solver instead of FW solver. To deal with pointwise input. Return [B,K] instead of [K]

    Step by step 
    Step 1: find solution for all input regardless constraints (i.e., (rest_weights - current_weight) * grads). Input [B, K, d], Return [B, K]
    Step 2: filter output with all positive constraints
    """
    vecs = [grads[:,ik,:] for ik in range(grads.shape[1])] # list of K vectors, each [B,d]
    sol = grad_descent_solv_batch(vecs, lr=1e-3, num_steps=100) # [B, K]
    

    print('sol', sol.shape)
    print('sol_0', sol[0].shape)

    print('w', w.shape)
    print('w_0', w[0].shape)

    """
    Equation 18, 
        alpha_i = lamda_i + sum_{all activated constraints} beta_i (u_ji - u_ki)
        k: current reference vector or 
    """
    # weight0 =  sol[0] + torch.sum(torch.stack([sol[j] * w[idx][j - 2 ,0] for j in torch.arange(2, 2 + torch.sum(idx))]))
    # weight1 =  sol[1] + torch.sum(torch.stack([sol[j] * w[idx][j - 2 ,1] for j in torch.arange(2, 2 + torch.sum(idx))]))
    # weight = torch.stack([weight0,weight1])
    
    # return weight

def test_get_d_paretomtl():
    K = 2 
    d = 10 
    B = 3 
    npref = 5 

    # grads = torch.randn(size=[K,d])
    # value = torch.randn(size=[K])
    # weights = torch.randn(size=[npref, K])
    # get_d_paretomtl_org(grads, value, weights, 0)

    grads = torch.randn(size=[B,K,d])
    value = torch.randn(size=[B,K])
    weights = torch.randn(size=[B,npref, K])

    get_d_paretomtl_pointwise(grads, value, weights, 0)

def ParetoEns(models, X, y, device, attack_params): 
    return pareto_ens(models, X, y, 
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

def pareto_ens(
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
    ParetoMTL to generate adversarial examples of ensemble of models.
    
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

    def _outer_max(delta):
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
        ParetoMTL Algorithm
        Description of the original algorithm in Multi-task learning 
            - Step 1: Collecting grads w.r.t. to each task's loss. 
                    Grad is on the shared params (encoder) only. 
            - Step 2: Create preference vectors. 
                The preference vectors is reinited in every epoch (in our setting is every iteration)
            - Step 3: Call paretomtl function to calculate the weights 
            - Step 4: normalizing weights 
        """
        # Parameters
        npref = 5 # number of preference vectors 
        pref_idx = int(gamma) # index in range(npref). using gamma variable for this purpose 
        assert(pref_idx >= 0 and pref_idx < npref)


        Gall = torch.stack(Gall, dim=2) # [B, CWH, K]
        
        ref_vec = torch.tensor(k_circle_points([1], [npref], K)[0]).cuda().float() # [npref, K]
        ref_vec = torch.repeat_interleave(torch.unsqueeze(ref_vec, dim=1), delta.shape[0], dim=1)  # [npref, B, K]
        assert (list(ref_vec.shape) == [npref, delta.shape[0], K])

        losses_vec = L

        weight_vec = get_d_paretomtl(Gall,losses_vec,ref_vec,pref_idx)


        normalize_coeff = K / torch.sum(torch.abs(weight_vec))
        weight_vec = weight_vec * normalize_coeff

        # Weighted the gradient 
        

        # normalize the gradients
        merged_grad = torch.reshape(merged_grad, shape=delta.shape)   

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




if __name__ == '__main__': 
    test_get_d_paretomtl()