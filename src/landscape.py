import torch
import torch.nn as nn
import numpy as np


class LandscapeAnalyzer:
    """
    Analyzes neural network loss landscape geometry
    Uses Hutchinson's method for efficient Hessian computations
    """

    def __init__(self, model, criterion, device, num_samples=100):
        self.model = model
        self.criterion = criterion
        self.device = device
        self.num_samples = num_samples

    def hessian_vector_product(self, inputs, targets, v):
        """
        Compute Hessian-vector product H*v efficiently
        """

        v = v.to(self.device)

        self.model.zero_grad()
        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)

        grads = torch.autograd.grad(
            loss, self.model.parameters(),
            create_graph=True, retain_graph=True
        )

        flat_grads = torch.cat([g.view(-1) for g in grads])

        flat_grads_v = (flat_grads * v).sum()

        hvp = torch.autograd.grad(
            flat_grads_v, self.model.parameters(),
            retain_graph=True
        )

        flat_hvp = torch.cat([h.view(-1) for h in hvp if h is not None])
        return flat_hvp
        
    def compute_trace(self, data_loader):
        """
        Compute Hessian trace using Hutchinson's method

        Trace = E[z^T H z] where z ~ Rademacher({-1, +1})
        Memory: O(n) instead of O(n²)
        """
        traces = []

        for batch_idx, (inputs, targets) in enumerate(data_loader):
            if batch_idx &gt;= 10:
                break

            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            params = list(self.model.parameters())
            total_params = sum(p.numel() for p in params)

            v = torch.randint(0, 2, (total_params,)).float().to(self.device)
            v = 2 * v - 1
            v = v / torch.norm(v)

            hvp = self.hessian_vector_product(inputs, targets, v)
            trace = (v * hvp).sum().item()
            traces.append(trace)

        return np.mean(traces) if traces else 0.0


    def compute_top_eigenvalue(self, data_loader, num_iter=5):
        """
        Estimate top Hessian eigenvalue using power iteration
        """
        params = list(self.model.parameters())
        total_params = sum(p.numel() for p in params)

        v = torch.randn(total_params).to(self.device)
        v = v / torch.norm(v)

        for iteration in range(num_iter):
            batch_x, batch_y = next(iter(data_loader))
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)


            hvp = self.hessian_vector_product(batch_x, batch_y, v)

            hvp_norm = torch.norm(hvp)
            if hvp_norm &gt; 1e-10:
                v = hvp / hvp_norm

        batch_x, batch_y = next(iter(data_loader))
        batch_x = batch_x.to(self.device)
        batch_y = batch_y.to(self.device)

        hvp = self.hessian_vector_product(batch_x, batch_y, v)
        eigenvalue = (hvp * v).sum().item()

        return abs(eigenvalue)

    def compute_metrics(self, data_loader):
        """
        Compute all landscape metrics
        """
        print(" Computing trace...")
        trace = self.compute_trace(data_loader)

        print(" Computing top eigenvalue...")
        top_eig = self.compute_top_eigenvalue(data_loader)

        min_eig = max(trace / 1000, 1e-6)
        condition_number = top_eig / min_eig if min_eig &gt; 0 else float('inf')

        return {
            'trace': trace,
            'top_eigenvalue': top_eig,
            'condition_number': condition_number,
            'flatness_measure': 1.0 / (1.0 + top_eig)
        }


