
import torch as ch
import numpy as np
from torch.distributions import laplace
from torch.distributions import uniform
def normalize_by_pnorm(x, p=2, small_constant=1e-6):
    """
    Normalize gradients for gradient (not gradient sign) attacks.
    # TODO: move this function to utils
    :param x: tensor containing the gradients on the input.
    :param p: (optional) order of the norm for the normalization (1 or 2).
    :param small_constant: (optional float) to avoid dividing by zero.
    :return: normalized gradients.
    """
    # loss is averaged over the batch so need to multiply the batch
    # size to find the actual gradient of each input sample

    assert isinstance(p, float) or isinstance(p, int)
    norm = _get_norm_batch(x, p)
    norm = ch.max(norm, ch.ones_like(norm) * small_constant)
    return batch_multiply(1. / norm, x)
def batch_l1_proj_flat(x, z=1):
    """
    Implementation of L1 ball projection from:
    :param x: input data
    :param eps: l1 radius
    :return: tensor containing the projection.
    """

    # Computing the l1 norm of v
    v = ch.abs(x)
    v = v.sum(dim=1)

    # Getting the elements to project in the batch
    indexes_b = ch.nonzero(v > z).view(-1)
    if isinstance(z, ch.Tensor):
        z = z[indexes_b][:, None]
    x_b = x[indexes_b]
    batch_size_b = x_b.size(0)

    # If all elements are in the l1-ball, return x
    if batch_size_b == 0:
        return x

    # make the projection on l1 ball for elements outside the ball
    view = x_b
    view_size = view.size(1)
    mu = view.abs().sort(1, descending=True)[0]
    vv = ch.arange(view_size).float().to(x.device)
    st = (mu.cumsum(1) - z) / (vv + 1)
    u = (mu - st) > 0
    if u.dtype.__str__() == "torch.bool":  # after and including ch 1.2
        rho = (~u).cumsum(dim=1).eq(0).sum(1) - 1
    else:  # before and including ch 1.1
        rho = (1 - u).cumsum(dim=1).eq(0).sum(1) - 1
    theta = st.gather(1, rho.unsqueeze(1))
    proj_x_b = _thresh_by_magnitude(theta, x_b)

    # gather all the projected batch
    proj_x = x.detach().clone()
    proj_x[indexes_b] = proj_x_b
    return proj_x

def _thresh_by_magnitude(theta, x):
    return ch.relu(ch.abs(x) - theta) * x.sign()

def batch_l1_proj(x, eps):
    batch_size = x.size(0)
    view = x.view(batch_size, -1)
    proj_flat = batch_l1_proj_flat(view, z=eps)
    return proj_flat.view_as(x)

def normalize_by_pnorm(x, p=2, small_constant=1e-6):
    """
    Normalize gradients for gradient (not gradient sign) attacks.
    # TODO: move this function to utils
    :param x: tensor containing the gradients on the input.
    :param p: (optional) order of the norm for the normalization (1 or 2).
    :param small_constant: (optional float) to avoid dividing by zero.
    :return: normalized gradients.
    """
    # loss is averaged over the batch so need to multiply the batch
    # size to find the actual gradient of each input sample

    assert isinstance(p, float) or isinstance(p, int)
    norm = _get_norm_batch(x, p)
    norm = ch.max(norm, ch.ones_like(norm) * small_constant)
    return batch_multiply(1. / norm, x)
def _get_norm_batch(x, p):
    batch_size = x.size(0)
    return x.abs().pow(p).view(batch_size, -1).sum(dim=1).pow(1. / p)
def batch_multiply(float_or_vector, tensor):
    if isinstance(float_or_vector, ch.Tensor):
        assert len(float_or_vector) == len(tensor)
        tensor = _batch_multiply_tensor_by_vector(float_or_vector, tensor)
    elif isinstance(float_or_vector, float):
        tensor *= float_or_vector
    else:
        raise TypeError("Value has to be float or torch.Tensor")
    return tensor
def _batch_multiply_tensor_by_vector(vector, batch_tensor):
    """Equivalent to the following
    for ii in range(len(vector)):
        batch_tensor[ii] *= vector[ii]
    return batch_tensor
    """
    return (
        batch_tensor.transpose(0, -1) * vector).transpose(0, -1).contiguous()
def rand_init_delta(delta, x, ord, eps, clip_min, clip_max):
    # TODO: Currently only considered one way of "uniform" sampling
    # for Linf, there are 3 ways:
    #   1) true uniform sampling by first calculate the rectangle then sample
    #   2) uniform in eps box then truncate using data domain (implemented)
    #   3) uniform sample in data domain then truncate with eps box
    # for L2, true uniform sampling is hard, since it requires uniform sampling
    #   inside a intersection of cube and ball, so there are 2 ways:
    #   1) uniform sample in the data domain, then truncate using the L2 ball
    #       (implemented)
    #   2) uniform sample in the L2 ball, then truncate using the data domain
    # for L1: uniform l1 ball init, then truncate using the data domain

    if isinstance(eps, ch.Tensor):
        assert len(eps) == len(delta)

    if ord == np.inf:
        delta.uniform_(-1, 1)
        delta = batch_multiply(eps, delta)
    elif ord == 2:
        delta.uniform_(clip_min, clip_max)
        delta = delta - x
        delta = clamp_by_pnorm(delta, ord, eps)
    elif ord == 1:
        ini = laplace.Laplace(
            loc=delta.new_tensor(0), scale=delta.new_tensor(1))
        delta = ini.sample(delta.shape)
        delta = normalize_by_pnorm(delta, p=1)
        ray = uniform.Uniform(0, eps).sample()
        delta *= ray
        delta = ch.clamp(x + delta, clip_min, clip_max) - x
    else:
        error = "Only ord = inf, ord = 1 and ord = 2 have been implemented"
        raise NotImplementedError(error)

    delta = ch.clamp(
        x + delta, min=clip_min, max=clip_max) - x
    return delta
def clamp_by_pnorm(x, p, r):
    assert isinstance(p, float) or isinstance(p, int)
    norm = _get_norm_batch(x, p)
    if isinstance(r, ch.Tensor):
        assert norm.size() == r.size()
    else:
        assert isinstance(r, float)
    factor = ch.min(r / norm, ch.ones_like(norm))
    return batch_multiply(factor, x)