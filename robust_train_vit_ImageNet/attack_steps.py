

import torch as ch
import numpy as np
from utils_l1 import *
from torch.distributions import laplace
from torch.distributions import uniform

class AttackerStep:
    '''
    Generic class for attacker steps, under perturbation constraints
    specified by an "origin input" and a perturbation magnitude.
    Must implement project, step, and random_perturb
    '''

    def __init__(self, orig_input, eps, step_size, use_grad=True):
        '''
        Initialize the attacker step with a given perturbation magnitude.
        Args:
            eps (float): the perturbation magnitude
            orig_input (ch.tensor): the original input
        '''
        self.orig_input = orig_input
        self.eps = eps
        self.step_size = step_size
        self.use_grad = use_grad

    def project(self, x):
        '''
        Given an input x, project it back into the feasible set
        Args:
            ch.tensor x : the input to project back into the feasible set.
        Returns:
            A `ch.tensor` that is the input projected back into
            the feasible set, that is,
        .. math:: \min_{x' \in S} \|x' - x\|_2
        '''
        raise NotImplementedError

    def step(self, x, g):
        '''
        Given a gradient, make the appropriate step according to the
        perturbation constraint (e.g. dual norm maximization for :math:`\ell_p`
        norms).
        Parameters:
            g (ch.tensor): the raw gradient
        Returns:
            The new input, a ch.tensor for the next step.
        '''
        raise NotImplementedError

    def random_perturb(self, x):
        '''
        Given a starting input, take a random step within the feasible set
        '''
        raise NotImplementedError

    def to_image(self, x):
        '''
        Given an input (which may be in an alternative parameterization),
        convert it to a valid image (this is implemented as the identity
        function by default as most of the time we use the pixel
        parameterization, but for alternative parameterizations this functino
        must be overriden).
        '''
        return x


### Instantiations of the AttackerStep class

# L-infinity threat model
class LinfStep(AttackerStep):
    """
    Attack step for :math:`\ell_\infty` threat model. Given :math:`x_0`
    and :math:`\epsilon`, the constraint set is given by:
    .. math:: S = \{x | \|x - x_0\|_\infty \leq \epsilon\}
    """

    def project(self, x):
        """
        """
        diff = x - self.orig_input
        #diff = ch.clamp(diff, -self.eps, self.eps)

        diff = ch.max(ch.min(diff, self.eps), -self.eps)

        cifar10_mean = (0.4914, 0.4822, 0.4465)
        cifar10_std = (0.2471, 0.2435, 0.2616)

        mu = ch.tensor(cifar10_mean).view(3, 1, 1).cuda()
        std = ch.tensor(cifar10_std).view(3, 1, 1).cuda()

        upper_limit = ((1 - mu) / std)
        lower_limit = ((0 - mu) / std)

        #out = ch.clamp(diff + self.orig_input, 0, 1)

        out = ch.max(ch.min(diff + self.orig_input, upper_limit), lower_limit)


        return out

    def step(self, x, g):
        """
        """
        step = ch.sign(g) * self.step_size
        return x + step

    def random_perturb(self, x):
        """
        """
        new_x = x + 2 * (ch.rand_like(x) - 0.5) * self.eps
        return ch.clamp(new_x, 0, 1)


# L2 threat model
class L2Step(AttackerStep):
    """
    Attack step for :math:`\ell_\infty` threat model. Given :math:`x_0`
    and :math:`\epsilon`, the constraint set is given by:
    .. math:: S = \{x | \|x - x_0\|_2 \leq \epsilon\}
    """

    def project(self, x):
        """
        """
        diff = x - self.orig_input
        diff = diff.renorm(p=2, dim=0, maxnorm=self.eps)
        return ch.clamp(self.orig_input + diff, 0, 1)

    def step(self, x, g):
        """
        """
        l = len(x.shape) - 1
        g_norm = ch.norm(g.view(g.shape[0], -1), dim=1).view(-1, *([1] * l))   # dim=1 compute do norm in one image
        scaled_g = g / (g_norm + 1e-10)
        return x + scaled_g * self.step_size

    def random_perturb(self, x):
        """
        """
        l = len(x.shape) - 1
        rp = ch.randn_like(x)
        rp_norm = rp.view(rp.shape[0], -1).norm(dim=1).view(-1, *([1] * l))  # first norm and then to be 4 dimensions
        return ch.clamp(x + self.eps * rp / (rp_norm + 1e-10), 0, 1)


# L1 threat model
# class L1Step(AttackerStep):
#     """
#     Attack step for :math:`\ell_\infty` threat model. Given :math:`x_0`
#     and :math:`\epsilon`, the constraint set is given by:
#     .. math:: S = \{x | \|x - x_0\|_2 \leq \epsilon\}
#     """
#
#     def project(self, x):
#         """
#         """
#         diff = x - self.orig_input
#         diff = diff.renorm(p=1, dim=0, maxnorm=self.eps)
#         return ch.clamp(self.orig_input + diff, 0, 1)
#
#     def step(self, x, g):
#         """
#         """
#         l = len(x.shape) - 1
#         # g_norm = ch.norm(g.view(g.shape[0], -1).abs(), dim=1).view(-1, *([1] * l))   # dim=1 compute do norm in one image
#         g_norm = ch.sum(g.view(g.shape[0], -1).abs(), dim=1).view(-1, *([1] * l))   # dim=1 compute do norm in one image
#         scaled_g = g / (g_norm + 1e-10)
#         return x + scaled_g * self.step_size
#
#     def random_perturb(self, x):
#         """
#         """
#         l = len(x.shape) - 1
#         rp = ch.randn_like(x)
#         rp_norm = rp.view(rp.shape[0], -1).abs().sum(dim=1).view(-1, *([1] * l))  # first norm and then to be 4 dimensions
#         return ch.clamp(x + self.eps * rp / (rp_norm + 1e-10), 0, 1)

# L1 threat model
class L1Step(AttackerStep):


    def project(self, x):
        """
        """
        diff = x - self.orig_input
        diff = batch_l1_proj(diff.cpu(), self.eps)
        if self.orig_input.is_cuda:
            diff = diff.cuda()
        return ch.clamp(self.orig_input + diff, 0.0, 1.0)

    def step(self, x, g, l1_sparsity=0.95):
        """
        modified from perturb_iterative function, in iterative_projected_gradient.py
        from advertorch on github
        """
        grad = g
        abs_grad = ch.abs(grad)
        batch_size = g.size(0)
        view = abs_grad.view(batch_size, -1)
        view_size = view.size(1)
        if l1_sparsity is None:
            vals, idx = view.topk(1)
        else:
            vals, idx = view.topk(int(np.round((1 - l1_sparsity) * view_size)))
        out = ch.zeros_like(view).scatter_(1, idx, vals)
        out = out.view_as(grad)
        grad = grad.sign() * (out > 0).float()

        grad = normalize_by_pnorm(grad, p=1)

        return x + grad * self.step_size

    def random_perturb(self, x):
        """
        modified from perturb function, in iterative_projected_gradient.py
        from advertorch on github
        """
        delta = ch.zeros_like(x)
        delta = ch.nn.Parameter(delta)
        # delta = rand_init_delta(delta, x, 1, self.eps, 0, 1)
        delta = rand_init_delta(delta=delta, x=x, ord=1, eps=self.eps, clip_min=0.0, clip_max=1.0)
        # delta = ch.clamp(x + delta, min=self.clip_min, max=self.clip_max) - x
        return ch.clamp(x + delta, 0, 1)

# Unconstrained threat model
class UnconstrainedStep(AttackerStep):
    """
    Unconstrained threat model, :math:`S = [0, 1]^n`.
    """

    def project(self, x):
        """
        """
        return ch.clamp(x, 0, 1)

    def step(self, x, g):
        """
        """
        return x + g * self.step_size

    def random_perturb(self, x):
        """
        """
        new_x = x + (ch.rand_like(x) - 0.5).renorm(p=2, dim=0, maxnorm=step_size)
        return ch.clamp(new_x, 0, 1)


class FourierStep(AttackerStep):

    def project(self, x):
        """
        """
        return x

    def step(self, x, g):
        """
        """
        return x + g * self.step_size

    def random_perturb(self, x):
        """
        """
        return x

    def to_image(self, x):
        """
        """
        return ch.sigmoid(ch.irfft(x, 2, normalized=True, onesided=False))


class RandomStep(AttackerStep):
    """
    Step for Randomized Smoothing.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_grad = False

    def project(self, x):
        """
        """
        return x

    def step(self, x, g):
        """
        """
        return x + self.step_size * ch.randn_like(x)

    def random_perturb(self, x):
        """
        """
        return x



    
