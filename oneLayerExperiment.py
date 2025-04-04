import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from timm.layers import MaskedLinear

import random
import numpy as np
from graphQualityMetric import get_mask_block_torch
import wandb

###############################################################################
# MaskedLinear Module
###############################################################################
""" class MaskedLinear(nn.Module):

    def __init__(self, in_features, out_features, bias=True, 
                 sparsity=0.8, block_size=2, device='cpu'):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        # Example: block mask
        mask = get_mask_block_torch((out_features, in_features), 
                                    sparsity=sparsity, 
                                    block_size=block_size, 
                                    device=device)
        
        # Register mask as a buffer (not learnable)
        self.register_buffer('mask', mask)

        # Permanently apply the mask on initialization
        with torch.no_grad():
            self.linear.weight.data.mul_(self.mask)

    def forward(self, x):
        w = self.linear.weight * self.mask
        return F.linear(x, w, self.linear.bias)

    def apply_mask(self):
        with torch.no_grad():
            self.linear.weight.data.mul_(self.mask) """

class MaskedLinearWithMask(nn.Module):
    """
    A linear layer that multiplies its weight by a (fixed) binary mask each forward pass.
    """
    def __init__(self, in_features, out_features, bias=True, sparsity=0.8, sparsityType='random', n=2,m=4,block_size=2 , mask=[], device='cuda'):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        # Ensure the mask is a tensor of the correct dtype
        if mask is None:
            # Example: If user didn't pass a mask, fill with ones
            mask_tensor = torch.ones_like(self.linear.weight.data)
        else:
            # If `mask` is a list or numpy array, convert it to a FloatTensor
            if not isinstance(mask, torch.Tensor):
                mask_tensor = torch.tensor(mask, dtype=torch.float32)
            else:
                mask_tensor = mask.float()
        
        mask_tensor = mask_tensor.to(self.linear.weight.device)
        # Register the mask as a buffer so PyTorch moves it to the right device with `model.to(device)`
        self.register_buffer('mask', mask_tensor)

        # Optionally, zero out the weights right away
        with torch.no_grad():
            self.linear.weight.data.mul_(self.mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Re-mask the weight each forward pass (in case of any updates)
        x = x.view(x.size(0), -1)
        w = self.linear.weight * self.mask
        return F.linear(x, w, self.linear.bias)

    def apply_mask(self):
        """Hard-apply the mask so the underlying weight stays zeroed out in masked entries."""
        with torch.no_grad():
            self.linear.weight.mul_(self.mask)

###############################################################################
# Simple One-Layer Network (784 -> 10)
###############################################################################
class OneLayerNet(nn.Module):
    def __init__(self, 
                 input_dim=784, 
                 num_classes=10, 
                 sparsity=0.8, 
                 sparsityType='random',
                 block_size=2, 
                 nm_n=2,
                 nm_m=2,
                 device='cpu'):
        super().__init__()
        self.fc = MaskedLinear(input_dim, num_classes,
                               sparsity=sparsity,
                               block_size=block_size,
                                sparsityType=sparsityType,
                                n=nm_n,
                                m=nm_m,
                                 bias=True,
                               device=device)

    def forward(self, x):
        # Flatten MNIST images from (B, 1, 28, 28) to (B, 784)
        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        return logits

class OneLayerLinearNet(nn.Module):
    def __init__(self, 
                 input_dim=784, 
                 num_classes=10, 
                 device='cpu'):
        super().__init__()
        self.fc =nn.Linear(input_dim, num_classes,
                               bias=True)
    def forward(self, x):
        # Flatten MNIST images from (B, 1, 28, 28) to (B, 784)
        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        return logits

###############################################################################
# Permutation Helper: W' = P W
# We'll permute the rows of W. That means constructing a permutation matrix P
# of size (out_features x out_features), and then do W' = P W. 
# Similarly we permute the mask the same way.
###############################################################################
def permute_rows(weight_or_mask, permutation):
    """
    weight_or_mask: (out_features, in_features)
    permutation: list or tensor of length out_features with a permutation of [0..out_features-1]
    Returns a new tensor W' = P W
    """
    # Re-order rows
    return weight_or_mask[permutation, :]

def create_permutation_matrix(num_rows):
    """
    Returns a random permutation of [0..num_rows-1]
    which we'll interpret as a row-reordering,
    seeded by the current system time.
    """
    import time
    import numpy as np
    
    # Seed with the current time
    np.random.seed(int(time.time() * 1e6)% (2**32 - 1))
    
    # Create a random permutation of indices
    perm = np.arange(num_rows)
    np.random.shuffle(perm)
    return perm

###############################################################################
# Training & Evaluation
###############################################################################
def train_one_epoch(model, dataloader, optimizer, device='cpu'):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for (data, target) in dataloader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        logits = model(data)
        loss = F.cross_entropy(logits, target)
        loss.backward()
        optimizer.step()
        # Re-apply mask so masked weights remain zero
        for module in model.modules():
            if isinstance(module, MaskedLinear):
                module.apply_mask()
        total_loss += loss.item()

        # If you want training accuracy, compute it here
        preds = torch.argmax(logits, dim=1)
        correct += (preds == target).sum().item()
        total += data.size(0)
    avg_loss = total_loss / len(dataloader)
    train_acc = correct / total if total > 0 else 0
    return avg_loss, train_acc

def evaluate(model, dataloader, device='cpu'):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for (data, target) in dataloader:
            data, target = data.to(device), target.to(device)
            logits = model(data)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == target).sum().item()
            total += data.size(0)
    return correct / total

###############################################################################
# Main Script
###############################################################################
def mainBlock():
    wandb.init(
        project="my-mnist-sparsity-project",  # Your W&B project name
        config={
            "batch_size": 256,
            "lr": 1e-2,
            "sparsityType": "block",
            "nm_n": 1,
            "nm_m": 20,
            "epochs": 150,
            "base_sparsity": 0.95,
            "block_size": 4,
            "dense": False,
        }
    )
    # Hyperparams
    config = wandb.config  # shorthand
    batch_size = config.batch_size
    lr = config.lr
    epochs = config.epochs
    base_sparsity = config.base_sparsity
    block_size = config.block_size
    sparsityType = config.sparsityType
    nm_n = config.nm_n
    nm_m = config.nm_m
    
    #name the run (using a vairable name) a combination of function name, sparsity type, block size, sparsity, and nm_n, nm_m
    wandb.run.name=f"block_sparsity_{block_size}_sparsity_{base_sparsity}_nm_n_{nm_n}_nm_m_{nm_m}",  # Name this run

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device:", device)
    
    # MNIST Data
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader   = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

    # 1) Train Base Model
    print("=== Training Base Model ===")

    if config.dense:
        base_model = OneLayerLinearNet(input_dim=784, num_classes=10, device=device).to(device)
    else:
        base_model = OneLayerNet(sparsity=base_sparsity, block_size=block_size, sparsityType=sparsityType, nm_n=nm_n, nm_m = nm_m, device=device).to(device)
    optimizer  = optim.SGD(base_model.parameters(), lr=lr)

    # Import CosineAnnealingLR and create scheduler
    from torch.optim.lr_scheduler import CosineAnnealingLR
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(base_model, train_loader, optimizer, device)
        val_acc = evaluate(base_model, test_loader, device)
        
        # Get current learning rate (CosineAnnealing might return multiple param groups)
        current_lr = scheduler.get_last_lr()[0]  # first param group for simplicity

        # Step the scheduler
        scheduler.step()

        # Log everything to wandb
        wandb.log({
            "Epoch": epoch + 1,
            "Train Loss": train_loss,
            "Train ACC": train_acc,
            "Validation ACC": val_acc,
            "Learning Rate": current_lr
        })

        print(f"Epoch {epoch+1}/{epochs} | Loss: {train_loss:.4f} | "
              f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | LR: {current_lr}")

    base_acc = evaluate(base_model, test_loader, device)
    print(f"Base Model Final Accuracy: {base_acc:.4f}")

    # Save the accuracy of the base model to a txt file
    with open('base_model_accuracy_block.txt', 'w') as f:
        f.write(f"Base Model Final Accuracy: {base_acc:.4f}\n")

    # We'll save the base mask and weights for reference
    base_mask = base_model.fc.mask.detach().clone()
    base_weight = base_model.fc.linear.weight.detach().clone()
    
    # 2) Generate 10 permutations, apply W' = P * W, and re-train
    results = []
    out_features = base_weight.shape[0]  # 10 for MNIST
    for perm_idx in range(100):
        print(f"\n=== Permutation {perm_idx} ===")
        #perm_model = OneLayerNet(sparsity=base_sparsity, block_size=block_size, sparsityType=sparsityType, nm_n=nm_n, nm_m = nm_m, device=device).to(device)

        perm_array = create_permutation_matrix(out_features)
        
        # Permute base weight and base mask
        perm_weight = permute_rows(base_weight, perm_array)
        perm_mask   = permute_rows(base_mask,   perm_array)

        perm_model = MaskedLinearWithMask(
            in_features=base_weight.shape[1],
            out_features=out_features,
            sparsity=base_sparsity,
            block_size=block_size,
            sparsityType=sparsityType,
            n=nm_n,
            m=nm_m,
            mask=perm_mask,  # Use the base mask for initialization
            device=device
        ).to(device)

        #with torch.no_grad():
            #perm_model.fc.linear.weight.copy_(perm_weight)
            #perm_model.fc.mask.copy_(perm_mask)

        perm_model.apply_mask()

        perm_optimizer = optim.SGD(perm_model.parameters(), lr=lr)
        perm_scheduler = CosineAnnealingLR(perm_optimizer, T_max=epochs)

        for epoch in range(epochs):
            train_loss, train_acc = train_one_epoch(perm_model, train_loader, perm_optimizer, device)
            val_acc = evaluate(perm_model, test_loader, device)
            current_lr = perm_scheduler.get_last_lr()[0]
            perm_scheduler.step()

            # Log some info for these permuted runs as well.
            # You might want a separate group/name or tag to keep them distinct in W&B.
            wandb.log({
                f"Permutation_{perm_idx}_Epoch": epoch + 1,
                f"Permutation_{perm_idx}_Train Loss": train_loss,
                f"Permutation_{perm_idx}_Train ACC": train_acc,
                f"Permutation_{perm_idx}_Val ACC": val_acc,
                f"Permutation_{perm_idx}_Learning Rate": current_lr
            })

            print(f"Permutation {perm_idx} | Epoch {epoch+1}/{epochs} "
                  f"| Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} "
                  f"| Val Acc: {val_acc:.4f} | LR: {current_lr}")

        final_acc = evaluate(perm_model, test_loader, device)
        print(f"Permutation {perm_idx} Final Accuracy: {final_acc:.4f}")

        # Grab the final weight after training
        final_weight = perm_model.linear.weight.detach().cpu().clone()

        results.append({
            'perm_idx': perm_idx,
            'permutation': perm_array,
            'final_accuracy': final_acc,
            'final_weight': final_weight  # <-- store it here
        })

        with open('permutation_results_block.txt', 'a') as f:
            f.write(f"Permutation {perm_idx} Final Accuracy: {final_acc:.4f}\n")
            f.write(f"Permutation Array: {perm_array}\n")
            f.write(f"Permutation Mask: {perm_mask}\n")
            f.write(f"Permutation Weight (initially permuted from base): {perm_weight}\n\n")
            f.write(f"** Final Trained Weight (after retraining) **\n{final_weight}\n\n")

    # 3) Summarize
    print("\n=== Summary ===")
    print(f"Base Model Acc: {base_acc:.4f}")
    for r in results:
        print(f"Perm {r['perm_idx']} -> Acc: {r['final_accuracy']:.4f}")

    # (Optional) Save everything
    torch.save({
         'base_model_state_dict': base_model.state_dict(),
         'base_accuracy': base_acc,
         'permutations': results
     }, 'block_sparsity_experiment_block.pth')

def mainDiag():
    wandb.init(
        project="my-mnist-sparsity-project",  # Your W&B project name
        #name="cosine-annealing-run",          # Name this run
        config={
            "batch_size": 256,
            "lr": 1e-2,
            "sparsityType": "diag",
            "nm_n": 1,
            "nm_m": 20,
            "epochs": 150,
            "base_sparsity": 0.95,
            "block_size": 4,
            "dense": False,
        }
    )

    # Hyperparams
    config = wandb.config  # shorthand
    batch_size = config.batch_size
    lr = config.lr
    epochs = config.epochs
    base_sparsity = config.base_sparsity
    block_size = config.block_size
    sparsityType = config.sparsityType
    nm_n = config.nm_n
    nm_m = config.nm_m
    
    #name the run (using a vairable name) a combination of function name, sparsity type, block size, sparsity, and nm_n, nm_m
    wandb.run.name=f"diag_block_sparsity_{block_size}_sparsity_{base_sparsity}_nm_n_{nm_n}_nm_m_{nm_m}",  # Name this run
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device:", device)

    # MNIST Data
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader   = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

    # 1) Train Base Model
    print("=== Training Base Model ===")

    if config.dense:
        base_model = OneLayerLinearNet(input_dim=784, num_classes=10, device=device).to(device)
    else:
        base_model = OneLayerNet(sparsity=base_sparsity, block_size=block_size, sparsityType=sparsityType, nm_n=nm_n, nm_m = nm_m, device=device).to(device)
    optimizer  = optim.SGD(base_model.parameters(), lr=lr)

    # Import CosineAnnealingLR and create scheduler
    from torch.optim.lr_scheduler import CosineAnnealingLR
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(base_model, train_loader, optimizer, device)
        val_acc = evaluate(base_model, test_loader, device)
        
        # Get current learning rate (CosineAnnealing might return multiple param groups)
        current_lr = scheduler.get_last_lr()[0]  # first param group for simplicity

        # Step the scheduler
        scheduler.step()

        # Log everything to wandb
        wandb.log({
            "Epoch": epoch + 1,
            "Train Loss": train_loss,
            "Train ACC": train_acc,
            "Validation ACC": val_acc,
            "Learning Rate": current_lr
        })

        print(f"Epoch {epoch+1}/{epochs} | Loss: {train_loss:.4f} | "
              f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | LR: {current_lr}")

    base_acc = evaluate(base_model, test_loader, device)
    print(f"Base Model Final Accuracy: {base_acc:.4f}")

    # Save the accuracy of the base model to a txt file
    with open('base_model_accuracy_diag.txt', 'w') as f:
        f.write(f"Base Model Final Accuracy: {base_acc:.4f}\n")

    # We'll save the base mask and weights for reference
    base_mask = base_model.fc.mask.detach().clone()
    base_weight = base_model.fc.linear.weight.detach().clone()
    
    # 2) Generate 10 permutations, apply W' = P * W, and re-train
    results = []
    out_features = base_weight.shape[0]  # 10 for MNIST
    for perm_idx in range(100):
        print(f"\n=== Permutation {perm_idx} ===")
        #perm_model = OneLayerNet(sparsity=base_sparsity, block_size=block_size, sparsityType=sparsityType, nm_n=nm_n, nm_m = nm_m, device=device).to(device)
        
        perm_array = create_permutation_matrix(out_features)
        
       # Permute base weight and base mask
        perm_weight = permute_rows(base_weight, perm_array)
        perm_mask   = permute_rows(base_mask,   perm_array)

        perm_model = MaskedLinearWithMask(
            in_features=base_weight.shape[1],
            out_features=out_features,
            sparsity=base_sparsity,
            block_size=block_size,
            sparsityType=sparsityType,
            n=nm_n,
            m=nm_m,
            mask=perm_mask,  # Use the base mask for initialization
            device=device
        ).to(device)

        #with torch.no_grad():
            #perm_model.fc.linear.weight.copy_(perm_weight)
            #perm_model.fc.mask.copy_(perm_mask)

        perm_model.apply_mask()
        perm_optimizer = optim.SGD(perm_model.parameters(), lr=lr)
        perm_scheduler = CosineAnnealingLR(perm_optimizer, T_max=epochs)

        for epoch in range(epochs):
            train_loss, train_acc = train_one_epoch(perm_model, train_loader, perm_optimizer, device)
            val_acc = evaluate(perm_model, test_loader, device)
            current_lr = perm_scheduler.get_last_lr()[0]
            perm_scheduler.step()

            # Log some info for these permuted runs as well.
            # You might want a separate group/name or tag to keep them distinct in W&B.
            wandb.log({
                f"Permutation_{perm_idx}_Epoch": epoch + 1,
                f"Permutation_{perm_idx}_Train Loss": train_loss,
                f"Permutation_{perm_idx}_Train ACC": train_acc,
                f"Permutation_{perm_idx}_Val ACC": val_acc,
                f"Permutation_{perm_idx}_Learning Rate": current_lr
            })

            print(f"Permutation {perm_idx} | Epoch {epoch+1}/{epochs} "
                  f"| Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} "
                  f"| Val Acc: {val_acc:.4f} | LR: {current_lr}")

        final_acc = evaluate(perm_model, test_loader, device)
        print(f"Permutation {perm_idx} Final Accuracy: {final_acc:.4f}")

       # Grab the final weight after training
        final_weight = perm_model.linear.weight.detach().cpu().clone()

        results.append({
            'perm_idx': perm_idx,
            'permutation': perm_array,
            'final_accuracy': final_acc,
            'final_weight': final_weight  # <-- store it here
        })

        with open('permutation_results_diag.txt', 'a') as f:
            f.write(f"Permutation {perm_idx} Final Accuracy: {final_acc:.4f}\n")
            f.write(f"Permutation Array: {perm_array}\n")
            f.write(f"Permutation Mask: {perm_mask}\n")
            f.write(f"Permutation Weight (initially permuted from base): {perm_weight}\n\n")
            f.write(f"** Final Trained Weight (after retraining) **\n{final_weight}\n\n")

    # 3) Summarize
    print("\n=== Summary ===")
    print(f"Base Model Acc: {base_acc:.4f}")
    for r in results:
        print(f"Perm {r['perm_idx']} -> Acc: {r['final_accuracy']:.4f}")

    # (Optional) Save everything
    torch.save({
         'base_model_state_dict': base_model.state_dict(),
         'base_accuracy': base_acc,
         'permutations': results
    }, 'block_sparsity_experiment_diag.pth')

def mainKM():
    wandb.init(
        project="my-mnist-sparsity-project",  # Your W&B project name
        name="cosine-annealing-run",          # Name this run
        config={
            "batch_size": 256,
            "lr": 1e-2,
            "sparsityType": "km",
            "nm_n": 1,
            "nm_m": 20,
            "epochs": 150,
            "base_sparsity": 0.95,
            "block_size": 4,
            "dense": False,
        }
    )

    # Hyperparams
    config = wandb.config  # shorthand
    batch_size = config.batch_size
    lr = config.lr
    epochs = config.epochs
    base_sparsity = config.base_sparsity
    block_size = config.block_size
    sparsityType = config.sparsityType
    nm_n = config.nm_n
    nm_m = config.nm_m
    #name the run (using a vairable name) a combination of function name, sparsity type, block size, sparsity, and nm_n, nm_m
    wandb.run.name=f"km_block_sparsity_{block_size}_sparsity_{base_sparsity}_nm_n_{nm_n}_nm_m_{nm_m}",  # Name this run
    

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device:", device)

    # MNIST Data
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader   = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

    # 1) Train Base Model
    print("=== Training Base Model ===")

    if config.dense:
        base_model = OneLayerLinearNet(input_dim=784, num_classes=10, device=device).to(device)
    else:
        base_model = OneLayerNet(sparsity=base_sparsity, block_size=block_size, sparsityType=sparsityType, nm_n=nm_n, nm_m = nm_m, device=device).to(device)
    optimizer  = optim.SGD(base_model.parameters(), lr=lr)

    # Import CosineAnnealingLR and create scheduler
    from torch.optim.lr_scheduler import CosineAnnealingLR
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(base_model, train_loader, optimizer, device)
        val_acc = evaluate(base_model, test_loader, device)
        
        # Get current learning rate (CosineAnnealing might return multiple param groups)
        current_lr = scheduler.get_last_lr()[0]  # first param group for simplicity

        # Step the scheduler
        scheduler.step()

        # Log everything to wandb
        wandb.log({
            "Epoch": epoch + 1,
            "Train Loss": train_loss,
            "Train ACC": train_acc,
            "Validation ACC": val_acc,
            "Learning Rate": current_lr
        })

        print(f"Epoch {epoch+1}/{epochs} | Loss: {train_loss:.4f} | "
              f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | LR: {current_lr}")

    base_acc = evaluate(base_model, test_loader, device)
    print(f"Base Model Final Accuracy: {base_acc:.4f}")

    # Save the accuracy of the base model to a txt file
    with open('base_model_accuracy_km.txt', 'w') as f:
        f.write(f"Base Model Final Accuracy: {base_acc:.4f}\n")

    # We'll save the base mask and weights for reference
    base_mask = base_model.fc.mask.detach().clone()
    base_weight = base_model.fc.linear.weight.detach().clone()
    
    # 2) Generate 10 permutations, apply W' = P * W, and re-train
    results = []
    out_features = base_weight.shape[0]  # 10 for MNIST
    for perm_idx in range(100):
        print(f"\n=== Permutation {perm_idx} ===")
        #perm_model = OneLayerNet(sparsity=base_sparsity, block_size=block_size, sparsityType=sparsityType, nm_n=nm_n, nm_m = nm_m, device=device).to(device)
        
        perm_array = create_permutation_matrix(out_features)
        
        # Permute base weight and base mask
        perm_weight = permute_rows(base_weight, perm_array)
        perm_mask   = permute_rows(base_mask,   perm_array)

        perm_model = MaskedLinearWithMask(
            in_features=base_weight.shape[1],
            out_features=out_features,
            sparsity=base_sparsity,
            block_size=block_size,
            sparsityType=sparsityType,
            n=nm_n,
            m=nm_m,
            mask=perm_mask,  # Use the base mask for initialization
            device=device
        ).to(device)

        #with torch.no_grad():
            #perm_model.fc.linear.weight.copy_(perm_weight)
            #perm_model.fc.mask.copy_(perm_mask)

        perm_model.apply_mask()

        perm_optimizer = optim.SGD(perm_model.parameters(), lr=lr)
        perm_scheduler = CosineAnnealingLR(perm_optimizer, T_max=epochs)

        for epoch in range(epochs):
            train_loss, train_acc = train_one_epoch(perm_model, train_loader, perm_optimizer, device)
            val_acc = evaluate(perm_model, test_loader, device)
            current_lr = perm_scheduler.get_last_lr()[0]
            perm_scheduler.step()

            # Log some info for these permuted runs as well.
            # You might want a separate group/name or tag to keep them distinct in W&B.
            wandb.log({
                f"Permutation_{perm_idx}_Epoch": epoch + 1,
                f"Permutation_{perm_idx}_Train Loss": train_loss,
                f"Permutation_{perm_idx}_Train ACC": train_acc,
                f"Permutation_{perm_idx}_Val ACC": val_acc,
                f"Permutation_{perm_idx}_Learning Rate": current_lr
            })

            print(f"Permutation {perm_idx} | Epoch {epoch+1}/{epochs} "
                  f"| Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} "
                  f"| Val Acc: {val_acc:.4f} | LR: {current_lr}")

        final_acc = evaluate(perm_model, test_loader, device)
        print(f"Permutation {perm_idx} Final Accuracy: {final_acc:.4f}")

        # Grab the final weight after training
        final_weight = perm_model.linear.weight.detach().cpu().clone()

        results.append({
            'perm_idx': perm_idx,
            'permutation': perm_array,
            'final_accuracy': final_acc,
            'final_weight': final_weight  # <-- store it here
        })

        with open('permutation_results_km.txt', 'a') as f:
            f.write(f"Permutation {perm_idx} Final Accuracy: {final_acc:.4f}\n")
            f.write(f"Permutation Array: {perm_array}\n")
            f.write(f"Permutation Mask: {perm_mask}\n")
            f.write(f"Permutation Weight (initially permuted from base): {perm_weight}\n\n")
            f.write(f"** Final Trained Weight (after retraining) **\n{final_weight}\n\n")

    # 3) Summarize
    print("\n=== Summary ===")
    print(f"Base Model Acc: {base_acc:.4f}")
    for r in results:
        print(f"Perm {r['perm_idx']} -> Acc: {r['final_accuracy']:.4f}")

     #(Optional) Save everything
    torch.save({
         'base_model_state_dict': base_model.state_dict(),
         'base_accuracy': base_acc,
         'permutations': results
     }, 'block_sparsity_experiment_km.pth')

def mainDense():
    wandb.init(
        project="my-mnist-sparsity-project",  # Your W&B project name
        config={
            "batch_size": 256,
            "lr": 1e-2,
            "sparsityType": "km",
            "nm_n": 1,
            "nm_m": 20,
            "epochs": 150,
            "base_sparsity": 0.95,
            "block_size": 4,
            "dense": True,
        }
    )

    # Hyperparams
    config = wandb.config  # shorthand
    batch_size = config.batch_size
    lr = config.lr
    epochs = config.epochs
    base_sparsity = config.base_sparsity
    block_size = config.block_size
    sparsityType = config.sparsityType
    nm_n = config.nm_n
    nm_m = config.nm_m

    #name the run (using a vairable name) a combination of function name, sparsity type, block size, sparsity, and nm_n, nm_m
    wandb.run.name = f"dense_sparsity_{base_sparsity}_nm_n_{nm_n}_nm_m_{nm_m}"  # Name this run

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device:", device)

    # MNIST Data
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader   = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

    # 1) Train Base Model
    print("=== Training Base Model ===")

    if config.dense:
        base_model = OneLayerLinearNet(input_dim=784, num_classes=10, device=device).to(device)
    else:
        base_model = OneLayerNet(sparsity=base_sparsity, block_size=block_size, sparsityType=sparsityType, nm_n=nm_n, nm_m = nm_m, device=device).to(device)
    
    #optimizer  = optim.SGD(base_model.parameters(), lr=lr)

    #Try adam optimizer with weight decay
    optimizer = optim.Adam(base_model.parameters(), lr=lr, weight_decay=1e-4)

    # Import CosineAnnealingLR and create scheduler
    from torch.optim.lr_scheduler import CosineAnnealingLR
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(base_model, train_loader, optimizer, device)
        val_acc = evaluate(base_model, test_loader, device)
        
        # Get current learning rate (CosineAnnealing might return multiple param groups)
        current_lr = scheduler.get_last_lr()[0]  # first param group for simplicity

        # Step the scheduler
        scheduler.step()

        # Log everything to wandb
        wandb.log({
            "Epoch": epoch + 1,
            "Train Loss": train_loss,
            "Train ACC": train_acc,
            "Validation ACC": val_acc,
            "Learning Rate": current_lr
        })

        print(f"Epoch {epoch+1}/{epochs} | Loss: {train_loss:.4f} | "
              f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | LR: {current_lr}")

    base_acc = evaluate(base_model, test_loader, device)
    print(f"Base Model Final Accuracy: {base_acc:.4f}")

    # Save the accuracy of the base model to a txt file
    with open('base_model_accuracy_km.txt', 'w') as f:
        f.write(f"Base Model Final Accuracy: {base_acc:.4f}\n")

    # We'll save the base mask and weights for reference
    base_mask = base_model.fc.mask.detach().clone()
    base_weight = base_model.fc.linear.weight.detach().clone()
    

    # 2) Summarize
    print("\n=== Summary ===")
    print(f"Base Model Acc: {base_acc:.4f}")
    for r in results:
        print(f"Perm {r['perm_idx']} -> Acc: {r['final_accuracy']:.4f}")

     #(Optional) Save everything
    torch.save({
         'base_model_state_dict': base_model.state_dict(),
         'base_accuracy': base_acc,
     }, 'dense.pth')

if __name__ == "__main__":
    mainDense()
    #mainBlock()
    #mainDiag()
    #mainKM()