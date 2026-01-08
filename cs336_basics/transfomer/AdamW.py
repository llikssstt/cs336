from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        defaults = {'lr':lr, 'betas':betas,'eps':eps, 'weight_decay':weight_decay}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group['lr']
            betas = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['m'] = torch.zeros_like(p.data)
                    state['v'] = torch.zeros_like(p.data)
                
                state['step'] += 1
                t = state['step']

                grad = p.grad.data
                m = state['m']
                v = state['v']

                m.mul_(betas[0]).add_(grad, alpha=1 - betas[0]) # m = beta1 * m + (1-beta1) * grad
                v.mul_(betas[1]).addcmul_(grad, grad, value=1 - betas[1]) # v = beta2 * v + (1-beta2) * grad^2
                # step_size = lr * sqrt(1 - beta2^t) / (1 - beta1^t)
                alpha_t = lr * math.sqrt(1 - betas[1]**t) / (1 - betas[0]**t)

                denom = v.sqrt().add_(eps)
                p.data.addcdiv_(m, denom, value=-alpha_t)

                if weight_decay > 0:
                    p.data.mul_(1 - lr * weight_decay)
        return loss