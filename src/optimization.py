import torch
import torch.nn as nn

class SAM(torch.optim.Optimizer):
    """
    Sharpness-Aware Minimization (SAM) Optimizer

    Paper: Foret et al. (2020)
    """

    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        self.base_optimizer = base_optimizer
        self.rho = rho
        self.param_groups = list(params) if isinstance(params, list) else params
        super().__init__(self.param_groups, {})

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        """First step: standard gradient descent"""
        self.state['original_params'] = []
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is not None:
                        self.state['original_params'].append(p.data.clone())
        grads = []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    grads.append(p.grad)

        grad_norm = torch.norm(torch.stack([torch.norm(g) for g in grads]))

        scale = self.rho / (grad_norm + 1e-12)
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                p.add_(p.grad, alpha=scale)

        if zero_grad:
            self.base_optimizer.zero_grad()

        @torch.no_grad()
        def second_step(self, zero_grad=False):
            """Second step: apply update from perturbed point"""
            for group in self.param_groups:
                for i, p in enumerate(group['params']):
                    if i &lt; len(self.state.get('original_params', [])):
                        p.data = self.state['original_params'][i]

            self.base_optimizer.step()

            if zero_grad:
                self.base_optimizer.zero_grad()



    def train_epoch(model, train_loader, criterion, optimizer, device):
        """Train for one epoch"""
        model.train()
        total_loss = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        return total_loss / len(train_loader)
    
    def validate(model, test_loader, criterion, device):
        """Validate on test set"""
        model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_x,batch_y in test_loader:
                batch_x,batch_y = batch_x.to(device), batch_y.to(device)

                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)

                total_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()

        accuracy = correct / total
        avg_loss = total_loss / len(test_loader)

        return accuracy, avg_loss
