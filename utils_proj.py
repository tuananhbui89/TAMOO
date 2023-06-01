import random
import numpy as np 
import torch 
import torch.nn.functional as F 
from torch.autograd import Variable

def normalize_grad(grad, norm, tol=1e-8):
    """
    Todo: normalizing gradient given a norm 
        Norm Linf: return sign(grad)
        Norm L1: 
        Norm L2: 
    """

    if norm == np.inf:
        grad = torch.sign(grad)
    elif norm == 1:
        ind = tuple(range(1, len(grad.shape)))
        grad = grad / (torch.sum(torch.abs(grad), dim=ind, keepdim=True) + tol)
    elif norm == 2:
        ind = tuple(range(1, len(grad.shape)))
        grad = grad / (torch.sqrt(torch.sum(torch.square(grad), dim=ind, keepdim=True)) + tol)
    
    return grad

def proj_box(v, p, eps, c, d):
    """
    TODO: Projecting into an appropriate box given type of norm 
    Args: 
        v: delta, input [B, C, H, W] # different with tf format 
        p: norm [np.inf, 1, 2]
        eps: 
        c: clip_min - x 
        d: clip_max - x
    """
    N, C, W, H = v.shape
    dim = W * H * C
    v = torch.reshape(v, (N, dim))
    c = torch.reshape(c, (N, dim))
    d = torch.reshape(d, (N, dim))

    clip_v = torch.minimum(v, d)
    clip_v = torch.maximum(clip_v, c)

    if p == np.inf:
        v = torch.clamp(clip_v, -eps, eps)

    elif p == 0:
        """
        """
        raise NotImplementedError
        e = torch.square(v)
        updates = torch.where(v < c, 1.0, 0.0).to(v.device)
        e = e - torch.multiply(updates, torch.square(v - c))
        updates = torch.where(d < v, 1.0, 0.0).to(v.device)
        e = e - torch.multiply(updates, torch.square(v - d))

        # Find int(eps)-th largest value in each row
        # if clip_v < n-th value --> 0 
        # if clip_v > n-th value --> 1  
        e_th, _ = torch.kthvalue(e, torch.int32(eps), dim=-1) # n-th element 
        v = torch.multiply(clip_v, 1 - torch.where(e < torch.stack([e_th] * dim, dim=1), 1.0, 0.0))

    elif p == 1:
        """
        Bisection method to find 1-norm of v 
        Left-boundary: c, Right-bounrdary: d 

        """
        def bi_norm1(v, c, d, max_iter=5):
            lam_l = torch.zeros(size=(N,)).to(v.device)
            lam_r = torch.max(torch.abs(v), dim=1)[0] - eps / dim

            for _ in range(max_iter): 
                lam = (lam_l + lam_r) / 2.0  
                eq = torch.norm(
                    torch.maximum(
                        torch.minimum(
                            torch.sign(v) * torch.relu(torch.abs(v) - torch.stack([lam] * dim, dim=1)), d), c
                        ),
                    p=1, 
                    dim=1
                ) - eps

                updates = torch.where(eq < 0.0, 1.0, 0.0).to(v.device)
                lam_r = lam_r - torch.multiply(updates, lam_r) + torch.multiply(updates, lam)

                updates = torch.where(eq > 0.0, 1.0, 0.0).to(v.device)
                lam_l = lam_l - torch.multiply(updates, lam_l) + torch.multiply(updates, lam)
            
            lam = (lam_l + lam_r) / 2.0 

            return torch.maximum(torch.minimum(torch.sign(v) * 
                                                torch.relu(torch.abs(v) - torch.stack([lam] * dim, dim=1)), d), c)

        """
        Todo: 
            - condition to use clip_v or bi_norm1 function. 
                if clip_v < eps: return clip_v  
                else: return bi_norm1(v, c, d)
        Note: 
        In the original implementation, they use reduce_all function --> logical all in the entire tensor --> 
        return True if all pixels < eps, else False. Using torch.all in replacement

        """
        v = torch.where(
            torch.all(torch.norm(clip_v, p=1, dim=1) < eps), clip_v, bi_norm1(v, c, d)
        )

    elif p == 2:
        """
        Bisection method to find 2-norm of v 

        """
        def bi_norm2(v, c, d, max_iter=5):
            lam_l = torch.zeros(size=(N,)).to(v.device)
            lam_r = torch.norm(clip_v, p=2, dim=1) / eps - 1

            for _ in range(max_iter): 
                lam = (lam_l + lam_r) / 2.0
                eq = torch.norm(
                    torch.maximum(
                        torch.minimum(
                            torch.divide(v, torch.stack([lam + 1] * dim, dim=1)), d
                        ), c
                    ), p=2, dim=1
                ) - eps 

                updates = torch.where(eq < 0.0, 1.0, 0.0).to(v.device)
                lam_r = lam_r - torch.multiply(updates, lam_r) + torch.multiply(updates, lam)

                updates = torch.where(eq > 0.0, 1.0, 0.0).to(v.device)
                lam_l = lam_l - torch.multiply(updates, lam_l) + torch.multiply(updates, lam)

            lam = (lam_l + lam_r) / 2.0 

            return torch.maximum(torch.minimum(torch.divide(v, torch.stack([lam + 1] * dim, dim=1)), d), c)

        """
        Todo: 
            - condition to use clip_v or bi_norm1 function. 
                if clip_v < eps: return clip_v  
                else: return bi_norm1(v, c, d)
        Note: 
        In the original implementation, they use reduce_all function --> logical all in the entire tensor --> 
        return True if all pixels < eps, else False. Using torch.all in replacement

        """

        v = torch.where(
            torch.all(torch.norm(clip_v, p=2, dim=1) < eps), clip_v, bi_norm2(v, c, d)
        )
    else:
        raise NotImplementedError("Values of `p` different from 0, 1, 2 and `np.inf` are currently not supported.")

    v = torch.reshape(v, [N, C, W, H])

    return v

def proj_box_appro(v, eps, p):
    """
    Project the values in `v` on the L_p norm ball of size `eps`.
    :param v: Array of perturbations to clip.
    :type v: `np.ndarray`
    :param eps: Maximum norm allowed.
    :type eps: `float`
    :param p: L_p norm to use for clipping. Only 1, 2 and `np.Inf` supported for now.
    :type p: `int`
    :return: Values of `v` after projection.
    :rtype: `np.ndarray`
    """

    # Pick a small scalar to avoid division by 0
    tol = 1e-8
    N, C, W, H = v.shape
    v_ = torch.reshape(v, (N, W * H * C))

    if p == 2:
        v_ = v_ * torch.unsqueeze(torch.minimum(1.0, eps / (torch.linalg.norm(v_, dim=1) + tol)), dim=1)
    elif p == 1:
        v_ = v_ * torch.unsqueeze(torch.minimum(1.0, eps / (torch.linalg.norm(v_, dim=1, ord=1) + tol)), dim=1)
    elif p == np.inf:
        # v_ = torch.sign(v_) * torch.minimum(torch.abs(v_), eps)
        v_ = torch.clamp(v_, min=-eps, max=eps)
    else:
        raise NotImplementedError("Values of `p` different from 1, 2 and `np.inf` are currently not supported.")

    v = torch.reshape(v_, [N, C, W, H])

    return v


def proj_prob_simplex(W, batch_size, K):
    W = bisection_mu(W, batch_size, K)
    return torch.maximum(torch.zeros(size=(batch_size, K)).to(W.device), W)


def bisection_mu(W, batch_size, K, max_iter=20):
    mu_l = torch.min(W, dim=1)[0] - 1 / K
    mu_r = torch.max(W, dim=1)[0] - 1 / K
    mu = (mu_l + mu_r) / 2.0
    eq = torch.sum(torch.relu(W - torch.stack([mu] * K, dim=1)), dim=1) - torch.ones(size=(batch_size,)).to(W.device)

    for _ in range(max_iter):
        mu = (mu_l + mu_r) / 2.0 
        eq = torch.sum(torch.relu(W - torch.stack([mu] * K, dim=1)), dim=1) - torch.ones(size=(batch_size,)).to(W.device)
        updates = torch.where(eq < 0.0, 1.0, 0.0).to(W.device)
        mu_r = mu_r - torch.multiply(updates, mu_r) + torch.multiply(updates, mu)
        updates = torch.where(0.0 < eq, 1.0, 0.0).to(W.device)
        mu_l = mu_l - torch.multiply(updates, mu_l) + torch.multiply(updates, mu)
    
    mu = (mu_l + mu_r) / 2.0 
    W -= torch.stack([mu] * K, dim=1)

    return W