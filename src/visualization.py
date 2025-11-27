import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


def plot_landscape_2d(model, criterion, data_loader, device, save_path,
grid_size=15, alpha=0.15):
    
    """Visualize 2D slice of loss landscape"""
    print(f" Creating 2D landscape plot...")
    params = []
    for p in model.parameters():
        params.append(p.data.view(-1))
    params_vec = torch.cat(params)

    d1 = torch.randn_like(params_vec)
    d1 = d1 / torch.norm(d1)
    d2 = torch.randn_like(params_vec)
    d2 = d2 / torch.norm(d2)

    def eval_loss(coeff1, coeff2):
        new_params = params_vec + alpha * coeff1 * d1 + alpha * coeff2 * d2

        offset = 0
        with torch.no_grad():
            for p in model.parameters():
                param_size = p.numel()
                p.copy_(new_params[offset:offset+param_size].view_as(p))
                offset += param_size


        model.eval()
        total_loss = 0
        count = 0
        with torch.no_grad():
            for batch_x, batch_y in data_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                total_loss += loss.item()
                count += 1

        return total_loss / count

    coords = np.linspace(-2, 2, grid_size)
    X, Y = np.meshgrid(coords, coords)
    Z = np.zeros_like(X, dtype=float)

    for i in range(grid_size):
        for j in range(grid_size):
            Z[i, j] = eval_loss(X[i, j], Y[i, j])

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap=cm.viridis, alpha=0.8)
    ax.set_xlabel('Direction 1')
    ax.set_ylabel('Direction 2')
    ax.set_zlabel('Loss')
    ax.set_title('Neural Network Loss Landscape (2D Slice)')

    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()

    print(f" Saved to {save_path}")

def plot_comparison(metrics_sgd, metrics_sam, save_path):

    """Compare metrics between SGD and SAM"""

    print(f" Creating comparison plot...")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('SGD vs SAM: Loss Landscape Metrics', fontsize=16)

    metrics = ['trace', 'top_eigenvalue', 'condition_number', 'flatness_measure']
    labels = ['Hessian Trace', 'Top Eigenvalue', 'Condition Number', 'Flatness']

    for idx, (metric, label) in enumerate(zip(metrics, labels)):
        ax = axes[idx // 2, idx % 2]

        sgd_val = metrics_sgd.get(metric, 0)
        sam_val = metrics_sam.get(metric, 0)

        x = np.arange(2)
        values = [sgd_val, sam_val]
        colors = ['#FF6B6B', '#4ECDC4']

        bars = ax.bar(x, values, color=colors, alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(['SGD', 'SAM'])
        ax.set_ylabel(label)
        ax.set_title(f'{label} Comparison')

        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}', ha='center', va='bottom')
            
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
    
    print(f"Saved to {save_path}")
