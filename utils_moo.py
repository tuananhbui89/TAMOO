from math import gamma
import numpy as np 
import torch 
from torch.autograd import Variable
import torch.optim as optim 
from min_norm_solvers import MinNormSolver

def gdquad_solv_single(vecs, lr, num_steps):
    """
        Gradient Descent Solver for Quadratic Optimization 
            C = argmin_c | c * vectors |_2^2 such that sum{c_i} = 1 and 1>c_i >0
        Args: 
            vecs: list of vectors, each vector must be 1D dimensions [D]
            lr: learning rate of solver 
            num_steps: number optimizations steps 
        Return: 
            C with shape [num_vecs]        
    """
    assert(type(vecs) is list) 
    assert(len(vecs[0].shape) == 1) 
    num_vecs = len(vecs)

    initw = 1/num_vecs * torch.ones(size=[num_vecs])
    lw = Variable(initw, requires_grad=True)
    opt = optim.Adam(params=[lw], lr=lr)
    
    for _ in range(num_steps): 
        softw = torch.softmax(lw, dim=0).to(vecs[0].device)
        with torch.enable_grad():
            sum_vec = 0 
            for vidx, vec in enumerate(vecs): 
                sum_vec += softw[vidx] * vec 
        
        norm_vec = torch.sum(torch.square(sum_vec))
        opt.zero_grad()
        norm_vec.backward()
        opt.step()
    
    finalw = Variable(softw.data, requires_grad=False).to(vecs[0].device)
    norm_vec = Variable(norm_vec.data, requires_grad=False).to(vecs[0].device)
    return finalw, norm_vec

def min_norm_element_from2(v1v1, v1v2, v2v2):
    """
    Analytical solution for min_{c} |cx_1 + (1-c)x_2|_2^2
    d is the distance (objective) optimzed
    v1v1 = <x1,x1>
    v1v2 = <x1,x2>
    v2v2 = <x2,x2>
    """
    if v1v2 >= v1v1:
        # Case: Fig 1, third column
        gamma = 0.999
        cost = v1v1
        return gamma, cost
    if v1v2 >= v2v2:
        # Case: Fig 1, first column
        gamma = 0.001
        cost = v2v2
        return gamma, cost
    # Case: Fig 1, second column
    gamma = -1.0 * ( (v1v2 - v2v2) / (v1v1+v2v2 - 2*v1v2) )
    cost = v2v2 + gamma*(v1v2 - v2v2)
    return gamma, cost

def quad_solv_for2D(vecs): 
    assert(type(vecs) is list) 
    assert(len(vecs[0].shape) == 1) 
    assert(len(vecs) == 2)
    v1v1 = torch.mul(vecs[0], vecs[0]).sum()
    v1v2 = torch.mul(vecs[0], vecs[1]).sum()
    v2v2 = torch.mul(vecs[1], vecs[1]).sum()

    gamma, _ = min_norm_element_from2(v1v1, v1v2, v2v2)
    finalw = torch.tensor([gamma, 1.0-gamma], requires_grad=False).to(vecs[0].device)
    return finalw 

def quad_solv_for2D_batch(vecs, output_mode='none'):
    """
        Solver for Quadratic Optimization. For 2D only
        Args: 
            vecs: list of vectors, each vector must be 2D dimensions [B, D]
        Return: 
            C with shape [batch_size, num_vecs] with num_vecs==2 
    """
    assert(type(vecs) is list) 
    assert(len(vecs[0].shape) == 2) 
    assert(len(vecs) == 2)
    batch_size = vecs[0].shape[0]
    num_vecs = len(vecs)

    v1v1 = torch.sum(torch.mul(vecs[0], vecs[0]), dim=1).data.cpu() # [B,]
    v1v2 = torch.sum(torch.mul(vecs[0], vecs[1]), dim=1).data.cpu() # [B,]
    v2v2 = torch.sum(torch.mul(vecs[1], vecs[1]), dim=1).data.cpu() # [B,]

    # gamma_mid = -1.0 * ( (v1v2 - v2v2) / (v1v1+v2v2 - 2*v1v2)) # [B,]
    # gamma_one = 0.999 * np.ones(shape=[batch_size]) # [B,]
    # gamma_zero = 0.001 * np.ones(shape=[batch_size]) # [B,]

    gamma = np.zeros(shape=[batch_size, num_vecs])
    cost = np.zeros(shape=[batch_size])
    for i in range(batch_size): 
        gamma[i,0], cost[i] = min_norm_element_from2(v1v1[i], v1v2[i], v2v2[i])
        gamma[i,1] = 1.0 - gamma[i,0]

    finalw = torch.tensor(gamma, requires_grad=False).to(vecs[0].device)

    if output_mode == 'none':
        return finalw
    elif output_mode == 'all': 
        return finalw, cost
    elif output_mode == 'mincost': 
        minidx = np.argmin(cost)
        return finalw[minidx,:]

def gdquad_solv_batch_slow(vecs, lr, num_steps, output_mode='none'):
    """
        Gradient Descent Solver for Quadratic Optimization 
            C = argmin_c | c * vectors |_2^2 such that sum{c_i} = 1 and 1>c_i >0
        Args: 
            vecs: list of vectors, each vector must be 2D dimensions [B, D]
            lr: learning rate of solver 
            num_steps: number optimizations steps 
        Return: 
            C with shape [batch_size, num_vecs]
        Note: 
            Each sample in batch must be independent, such that solution C is 
            correct for every sample. Therefore, cannot minimize the loss of whole 
            batch 
            In this slow version, will be separately solving MOO for each sample 
        Unsolve: 
        - How to deal with batch_norm layer? Answer: there is no batchnorm layer here 
    """
    assert(type(vecs) is list) 
    assert(len(vecs[0].shape) == 2) 
    num_vecs = len(vecs)
    batch_size, d = vecs[0].shape

    batchw = []
    batchcost = []
    for isample in range(batch_size):
        _vecs = [vec[isample,:] for vec in vecs] # list of 1D vector 
        onew, onecost = gdquad_solv_single(_vecs, lr, num_steps)
        onew = torch.unsqueeze(onew, dim=0)
        onecost = torch.unsqueeze(onecost, dim=0)
        batchw.append(onew)
        batchcost.append(onecost)

    batchw = torch.cat(batchw, dim=0)
    batchw = Variable(batchw.data, requires_grad=False)
    batchcost = torch.cat(batchcost, dim=0)
    batchcost = Variable(batchcost.data, requires_grad=False)

    if output_mode == 'all':
        return batchw, batchcost
    elif output_mode == 'mincost': 
        minidx = torch.argmin(batchcost)
        return batchw[minidx,:]
    elif output_mode == 'none': 
        return batchw

def grad_descent_solv_batch(vecs, lr, num_steps):
    """
        Gradient Descent Solver for Quadratic Optimization 
            C = argmin_c | c * vectors |_2^2 such that sum{c_i} = 1 and 1>c_i >0
        Args: 
            vecs: list of vectors, each vector must be 2D dimensions [B, D]
            lr: learning rate of solver 
            num_steps: number optimizations steps 
        Return: 
            C with shape [batch_size, num_vecs]
        Note: 
            Each sample in batch must be independent, such that solution C is 
            correct for every sample. Therefore, cannot minimize the loss of whole 
            batch 
        
        Unsolve: 
        - How to deal with batch_norm layer? Answer: there is no batchnorm layer here 
    """
    assert(type(vecs) is list) 
    assert(len(vecs[0].shape) == 2) 
    num_vecs = len(vecs)
    batch_size, d = vecs[0].shape

    initw = 1/num_vecs * torch.ones(size=[batch_size, num_vecs])
    # initw = torch.rand(size=[batch_size, num_vecs])
    lw = Variable(initw, requires_grad=True)
    opt = optim.Adam(params=[lw], lr=lr)
    

    for step in range(num_steps): 
        softw = torch.softmax(lw, dim=0)

        with torch.enable_grad():
            sum_vec = 0 
            for vidx, vec in enumerate(vecs): 
                sum_vec += torch.unsqueeze(softw[:, vidx], dim=1) * vec # [batch_size, 1] * [batch_size, d]  
        
        norm_vec = torch.sum(torch.square(sum_vec), dim=1)
        opt.zero_grad()
        # norm_vec.backward(gradient=torch.ones_like(norm_vec))
        # opt.step()
        # grad = torch.autograd.grad(outputs=norm_vec, inputs=lw, grad_outputs=torch.ones_like(norm_vec))[0]
        # print(grad.shape)

        # lw = lw - lr * grad

        # if step % 10 == 0: 
        #     print('step={}, norm_vec={}, softw={}'.format(step, norm_vec.detach().cpu().numpy(), softw.detach().cpu().numpy()))
    
    finalw = Variable(softw.data, requires_grad=False)
    return finalw

def test_grad_descent_solver(): 
    """
    Create three scenarios with 2D vectors and test quadratic solvers 
    - Using Gradient descent method 
    - Using closed form method 
    """

    v1 = [[-5, -5], [5, 1], [-5, -5]]
    v2 = [[-5, -1], [5, 5], [5, 5]] 
    v3 = [[-1, 1], [1, 1], [5, 0]]

    tv1 = torch.tensor(np.asarray(v1))
    tv2 = torch.tensor(np.asarray(v2))
    tv3 = torch.tensor(np.asarray(v3))

    batchw, batchcost = gdquad_solv_batch_slow([tv1, tv2, tv3], lr=0.5, num_steps=100, output_mode='all')
    print('gdquad_solv_batch_slow', batchw[0,:], 'batchcost', batchcost[0])
    print('gdquad_solv_batch_slow', batchw[1,:], 'batchcost', batchcost[1])
    print('gdquad_solv_batch_slow', batchw[2,:], 'batchcost', batchcost[2])
    print('-----------------------')
    singlew1, cost1 = gdquad_solv_single([tv1[0,:], tv2[0,:], tv3[0, :]], lr=0.1, num_steps=100)
    singlew2, cost2 = gdquad_solv_single([tv1[1,:], tv2[1,:], tv3[1, :]], lr=0.1, num_steps=100)
    singlew3, cost3 = gdquad_solv_single([tv1[2,:], tv2[2,:], tv3[2, :]], lr=0.1, num_steps=100)
    print('gdquad_solv_single', singlew1, 'cost1', cost1)
    print('gdquad_solv_single', singlew2, 'cost2', cost2)
    print('gdquad_solv_single', singlew3, 'cost3', cost3)
    print('-----------------------')
    closefw1 = quad_solv_for2D([tv1[0,:], tv2[0,:]])
    closefw2 = quad_solv_for2D([tv1[1,:], tv2[1,:]])
    closefw3 = quad_solv_for2D([tv1[2,:], tv2[2,:]])
    print('quad_solv_for2D', closefw1)
    print('quad_solv_for2D', closefw2)
    print('quad_solv_for2D', closefw3)
    print('-----------------------')
    closefw = quad_solv_for2D_batch([tv1, tv2])
    print('quad_solv_for2D_batch', closefw[0,:])
    print('quad_solv_for2D_batch', closefw[1,:])
    print('quad_solv_for2D_batch', closefw[2,:])

    print('-----------------------')
    mgds, cost = MinNormSolver.find_min_norm_element([tv1, tv2, tv3])
    print('find_min_norm_element', mgds, 'cost', cost)
    print(type(mgds))

    # print('-----------------------')
    # mgds, cost = MinNormSolver.find_min_norm_element_FW([tv1, tv2, tv3])
    # print('find_min_norm_element_FW', mgds, 'cost', cost)

if __name__ == "__main__":
    test_grad_descent_solver()
