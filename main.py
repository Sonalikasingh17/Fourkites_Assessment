"""
Loss Landscape Analysis Framework

In this file:
1. Trains models using both SGD and SAM optimizers
2. Analyzes their loss landscapes
3. Computes key metrics (Hessian, eigenvalues, curvature)
4. Creates visualizations
5. Generates comparison reports
"""

import argparse
import torch
import torch.nn as nn
import numpy as np
import yaml
from pathlib import Path
from tqdm import tqdm

# Import our modules
from src.utils import set_seed, load_data, save_checkpoint, load_checkpoint
from src.model import SimpleMLP, SimpleCNN
from src.optimization import SAM, train_epoch, validate
from src.landscape import LandscapeAnalyzer
from src.visualization import plot_landscape_2d, plot_comparison


def main(args):
    """
    Main pipeline orchestrator
    
    Flow:
    1. Setup: seed, device, directories
    2. Load: data, model
    3. Train: using SGD and SAM
    4. Analyze: compute Hessian, eigenvalues
    5. Visualize: create 2D plots
    6. Report: save results
    """
    
    # ========== SETUP ==========
    print("=" * 60)
    print("LOSS LANDSCAPE ANALYSIS FRAMEWORK")
    print("=" * 60)
    
    # Set reproducibility
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # ========== DATA ==========
    print(f"\nLoading {args.dataset} dataset...")
    train_loader, test_loader, input_size, num_classes = load_data(
        args.dataset, args.batch_size
    )
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # ========== MODEL SELECTION ==========
    print(f"\nCreating model: {args.model}")
    if args.model == "mlp":
        model = SimpleMLP(input_size, num_classes, hidden_dims=args.hidden_dims)
    else:
        model = SimpleCNN(num_classes)
    
    model = model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # ========== LOSS FUNCTION ==========
    criterion = nn.CrossEntropyLoss()
    
    # ========== TRAIN WITH SGD ==========
    print("\n" + "=" * 60)
    print("TRAINING WITH SGD")
    print("=" * 60)
    
    model_sgd = model.__class__(*[input_size, num_classes] if args.model == "mlp" 
                                else [num_classes]).to(device)
    optimizer_sgd = torch.optim.SGD(
        model_sgd.parameters(), 
        lr=args.lr, 
        momentum=0.9
    )
    
    best_acc_sgd = 0
    for epoch in range(args.epochs):
        train_loss = train_epoch(
            model_sgd, train_loader, criterion, optimizer_sgd, device
        )
        test_acc, test_loss = validate(
            model_sgd, test_loader, criterion, device
        )
        
        if test_acc > best_acc_sgd:
            best_acc_sgd = test_acc
            save_checkpoint(model_sgd, output_dir / "model_sgd_best.pt")
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{args.epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Test Acc: {test_acc:.4f}")
    
    print(f"Best SGD Test Accuracy: {best_acc_sgd:.4f}")
    
    # ========== TRAIN WITH SAM ==========
    print("\n" + "=" * 60)
    print("TRAINING WITH SAM (Sharpness-Aware Minimization)")
    print("=" * 60)
    
    model_sam = model.__class__(*[input_size, num_classes] if args.model == "mlp" 
                                 else [num_classes]).to(device)
    base_optimizer = torch.optim.SGD
    optimizer_sam = SAM(
        model_sam.parameters(),
        base_optimizer,
        lr=args.lr,
        momentum=0.9,
        rho=2.0  # SAM rho parameter
    )
    
    best_acc_sam = 0
    for epoch in range(args.epochs):
        # SAM training
        model_sam.train()
        for batch_x, batch_y in tqdm(train_loader, disable=True):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            # First forward-backward pass
            loss1 = criterion(model_sam(batch_x), batch_y)
            loss1.backward()
            optimizer_sam.first_step(zero_grad=True)
            
            # Second forward-backward pass
            loss2 = criterion(model_sam(batch_x), batch_y)
            loss2.backward()
            optimizer_sam.second_step(zero_grad=True)
        
        test_acc, test_loss = validate(
            model_sam, test_loader, criterion, device
        )
        
        if test_acc > best_acc_sam:
            best_acc_sam = test_acc
            save_checkpoint(model_sam, output_dir / "model_sam_best.pt")
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{args.epochs} | "
                  f"Test Acc: {test_acc:.4f}")
    
    print(f"Best SAM Test Accuracy: {best_acc_sam:.4f}")
    print(f"Improvement: {(best_acc_sam - best_acc_sgd):.4f}")
    
    # ========== LANDSCAPE ANALYSIS ==========
    print("\n" + "=" * 60)
    print("ANALYZING LOSS LANDSCAPES")
    print("=" * 60)
    
    analyzer_sgd = LandscapeAnalyzer(model_sgd, criterion, device)
    analyzer_sam = LandscapeAnalyzer(model_sam, criterion, device)
    
    print("Computing Hessian for SGD model...")
    metrics_sgd = analyzer_sgd.compute_metrics(test_loader)
    
    print("Computing Hessian for SAM model...")
    metrics_sam = analyzer_sam.compute_metrics(test_loader)
    
    # Print metrics
    print("\n" + "-" * 60)
    print("METRICS COMPARISON")
    print("-" * 60)
    print(f"{'Metric':<20} {'SGD':<20} {'SAM':<20}")
    print("-" * 60)
    for key in metrics_sgd.keys():
        print(f"{key:<20} {metrics_sgd[key]:<20.6f} {metrics_sam[key]:<20.6f}")
    
    # ========== VISUALIZATION ==========
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)
    
    # 2D Landscape plots
    print("Creating 2D landscape visualizations...")
    plot_landscape_2d(
        model_sgd, criterion, test_loader, device,
        output_dir / "landscape_sgd.png"
    )
    plot_landscape_2d(
        model_sam, criterion, test_loader, device,
        output_dir / "landscape_sam.png"
    )
    
    # Comparison plots
    print("Creating comparison plots...")
    plot_comparison(
        metrics_sgd, metrics_sam,
        output_dir / "metrics_comparison.png"
    )
    
    # ========== SAVE RESULTS ==========
    print("\n" + "=" * 60)
    print("RESULTS SAVED")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print("Files generated:")
    print("  - model_sgd_best.pt (trained model)")
    print("  - model_sam_best.pt (trained model)")
    print("  - landscape_sgd.png (SGD loss landscape)")
    print("  - landscape_sam.png (SAM loss landscape)")
    print("  - metrics_comparison.png (metrics comparison)")
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"SGD Test Accuracy: {best_acc_sgd:.4f}")
    print(f"SAM Test Accuracy: {best_acc_sam:.4f}")
    print(f"Improvement: {((best_acc_sam - best_acc_sgd) / best_acc_sgd * 100):.2f}%")
    print(f"\nSGD Sharpness (top eigenvalue): {metrics_sgd['top_eigenvalue']:.4f}")
    print(f"SAM Sharpness (top eigenvalue): {metrics_sam['top_eigenvalue']:.4f}")
    print(f"Sharpness Reduction: {((metrics_sgd['top_eigenvalue'] - metrics_sam['top_eigenvalue']) / metrics_sgd['top_eigenvalue'] * 100):.2f}%")
    print("\n✓ Analysis complete! Check visualizations in output directory.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Loss Landscape Geometry & Optimization Dynamics"
    )
    
    # Dataset & Model
    parser.add_argument("--dataset", type=str, default="mnist", 
                       choices=["mnist", "cifar10"])
    parser.add_argument("--model", type=str, default="mlp", 
                       choices=["mlp", "cnn"])
    parser.add_argument("--hidden_dims", type=list, default=[128, 64])
    
    # Training
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    
    # Other
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="./results")
    
    args = parser.parse_args()
    main(args)
