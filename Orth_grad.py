import torch
import math
from torch.optim import Optimizer

class Orth_grad(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), alpha=0.1, eps=1e-8, weight_decay=0):
        """
        Args:
            lr (float): learning rate (default: 1e-3)
            betas (Tuple[float, float]): momentum params (β1, β2) (default: (0.9, 0.999))
            alpha (float): Orthogonal gradient params (default: 0.1)
            eps (float):  (default: 1e-8)
            weight_decay (float): weight decay (L2 punishment) (default: 0)
        """
        defaults = dict(lr=lr, betas=betas, alpha=alpha, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @staticmethod
    def newtonschulz5(G, steps=5, eps=1e-7):
        """
        Args:
            G (Tensor): Input (2D)
            steps (int):  (default: 5)
            eps (float)
        Returns:
            Tensor: Orthogonal Matrix
        """
        assert G.ndim == 2
        a, b, c = (3.4445, -4.7750, 2.0315)  # parameters from Muon

        # Normalization
        X = G / (torch.norm(G, p='fro') + eps)
        transpose_needed = G.size(0) > G.size(1)
        if transpose_needed:
            X = X.T

        # Newton-Schulz
        for _ in range(steps):
            A = X @ X.T
            B = b * A + c * A @ A
            X = a * X + B @ X

        return X.T if transpose_needed else X

    def orthogonalize_grad(self, g_orth, grad, curv=None, eps=1e-7):
        """
        Args:
            g_orth (Tensor): orthogonal gradient
            grad (Tensor): gradient
            curv (Tensor): curvature
            eps (float)
        Returns:
            Tensor: Orthogonal Matrix
        """
        # 合并梯度和曲率信息
        if curv is not None and curv.norm() > eps:
            combined = torch.stack([grad.flatten(), curv.flatten()])  # [2, d]
        else:
            combined = grad.flatten().unsqueeze(0)  # [1, d]

        # applying Newton-Schulz
        orth_matrix = self.newtonschulz5(combined, steps=5)
        return orth_matrix[0].view_as(g_orth)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Refuse Sparse Gradient')

                state = self.state.setdefault(p, {})

                # Initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['m'] = torch.zeros_like(p.data)
                    state['v'] = torch.zeros_like(p.data)
                    state['g_orth'] = torch.zeros_like(p.data)
                    state['prev_grad'] = torch.zeros_like(p.data)

                # Weight Decay
                if group['weight_decay'] != 0:
                    grad = grad.add(p.data, alpha=group['weight_decay'])

                # Curvature Estimation
                curv = grad - state['prev_grad']
                state['prev_grad'].copy_(grad)

                # Adam Part
                beta1, beta2 = group['betas']
                state['step'] += 1
                state['m'].mul_(beta1).add_(grad, alpha=1 - beta1)
                state['v'].mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                bias_corr1 = 1 - beta1 ** state['step']
                bias_corr2 = 1 - beta2 ** state['step']
                m_hat = state['m'] / bias_corr1
                v_hat = state['v'] / bias_corr2

                # calling orthogonalize
                state['g_orth'] = self.orthogonalize_grad(
                    state['g_orth'],
                    grad,
                    curv if curv.norm() > group['eps'] else None,
                    eps=group['eps']
                )

                # params updating
                effective_alpha = group['alpha'] * math.exp(-0.005 * state['step'])
                denom = v_hat.sqrt().add_(group['eps'])
                update = m_hat / denom + effective_alpha * state['g_orth']
                p.data.add_(update, alpha=-group['lr'])


        return loss
