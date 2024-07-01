
from attack_steps import L2Step, LinfStep, L1Step
import torch
import torch.nn as nn
import numpy as np
import torchvision
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import utils

# from utils import std, mu

num_classes = 10


class AttackPGD(nn.Module):  # change here to build l_2 and l_inf
    def __init__(self, basic_net, config):
        super(AttackPGD, self).__init__()
        self.basic_net = basic_net
        self.rand = config['random_start']
        self.step_size = config['step_size']
        self.epsilon = config['epsilon']
        self.num_steps = config['num_steps']
        self._type = config['_type']
        assert config['loss_func'] == 'xent', 'Only xent supported for now.'
        self.targeted = False
        self.random_start = self.rand

    def forward(self, inputs, targets, attack, y_targets=None):
        if not attack:
            return self.basic_net(inputs), inputs
        step = None
        if self._type == 'l2':
            step = L2Step(eps=self.epsilon, orig_input=inputs, step_size=self.step_size)
        elif self._type == 'linf':
            step = LinfStep(eps=self.epsilon, orig_input=inputs, step_size=self.step_size)
        elif self._type == 'l1':
            step = L1Step(eps=self.epsilon, orig_input=inputs, step_size=self.step_size)
        else:
            NotImplementedError
        x = inputs.clone().detach().requires_grad_(True)

        # Random start (to escape certain types of gradient masking)
        if self.random_start:
            x = step.random_perturb(x)
        criterion = torch.nn.CrossEntropyLoss(reduction='none')
        m = -1 if self.targeted else 1
        for _ in range(self.num_steps):
            x = x.clone().detach().requires_grad_(True)
            outputs = self.basic_net(x)
            losses = criterion(outputs, targets if not self.targeted else y_targets)
            assert losses.shape[0] == x.shape[0], 'shape of losses needs to match input'
            loss = torch.mean(losses)
            grad, = torch.autograd.grad(m * loss, [x])
            with torch.no_grad():
                x = step.step(x, grad)
                x = step.project(x)
        ret = x.clone().detach()
        return ret


# def project(x, original_x, epsilon, _type='linf'):
#
#     if _type == 'linf':
#         max_x = original_x + epsilon
#         min_x = original_x - epsilon
#
#         x = torch.max(torch.min(x, max_x), min_x)
#
#     elif _type == 'l2':
#         dist = (x - original_x)
#
#         dist = dist.view(x.shape[0], -1)
#
#         dist_norm = torch.norm(dist, dim=1, keepdim=True)
#
#         mask = (dist_norm > epsilon).unsqueeze(2).unsqueeze(3)
#
#         # dist = F.normalize(dist, p=2, dim=1)
#
#         dist = dist / dist_norm
#
#         dist *= epsilon
#
#         dist = dist.view(x.shape)
#
#         x = (original_x + dist) * mask.float() + x * (1 - mask.float())
#
#
#     return x
#
#



# Model
# class AttackPGD(nn.Module):  # change here to build l_2 and l_inf
#     def __init__(self, basic_net, config):
#         super(AttackPGD, self).__init__()
#         self.basic_net = basic_net
#         self.rand = config['random_start']
#         self.step_size = config['step_size']
#         self.epsilon = config['epsilon']
#         self.num_steps = config['num_steps']
#         self._type = config['_type']
#         assert config['loss_func'] == 'xent', 'Only xent supported for now.'
#
#     def forward(self, inputs, targets, attack):
#         if not attack:
#             return self.basic_net(inputs), inputs
#
#         x = inputs.detach()
#         if self.rand:     # attack here l_inf attack
#             x = x + torch.zeros_like(x).uniform_(-self.epsilon, self.epsilon)
#         for i in range(self.num_steps):
#             x.requires_grad_()
#             with torch.enable_grad():
#                 logits = self.basic_net(x)
#                 #loss = F.cross_entropy(logits, targets, size_average=False)
#                 loss = F.cross_entropy(logits, targets, reduction='sum')
#             grad = torch.autograd.grad(loss, [x])[0]
#             x = x.detach() + self.step_size*torch.sign(grad.detach())
#
#            # normal l_infite
#            # x = torch.min(torch.max(x, inputs - self.epsilon), inputs + self.epsilon)
#
#             x = project(x, inputs, self.epsilon, self._type)
#             x = torch.clamp(x, 0, 1)
#
#         #return self.basic_net(x), x
#         return x




    
