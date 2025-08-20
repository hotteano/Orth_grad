import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.optim import Optimizer
import math
import numpy as np

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        return orth_matrix[0].view_as(g_orth)  # 取首行恢复形状

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

                # 初始化状态
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

class CIFARNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25)
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} '
          f'({100. * correct / len(test_loader.dataset):.2f}%)\n')

def check_orthogonality(model, optimizer, test_loader):
    model.eval()
    orth_scores = []

    for data, _ in test_loader:
        data = data.to(device)

        output = model(data)

        target = torch.randint(0, 10, (data.size(0),), device=device)
        loss = F.cross_entropy(output, target)

        model.zero_grad()
        loss.backward()


        for p in model.parameters():
            if p.grad is not None and p.requires_grad:
                grad = p.grad.data
                m_orth = optimizer.state[p]['m_orth']
                dot = torch.dot(grad.flatten(), m_orth.flatten()).item()
                orth_scores.append(abs(dot))

    print(f"Orthogonality score: {np.mean(orth_scores):.2e} (should be close to 0)")

def main():
    # 增强的数据转换
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # 数据加载
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=128, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=256, shuffle=False, num_workers=4)
    # 模型和优化器
    model = CIFARNet().to(device)
    optimizer = Orth_grad(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

    # 训练循环
    for epoch in range(1, 10):  # 延长训练周期
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    check_orthogonality(model, optimizer, test_loader)


if __name__ == '__main__':
    main()