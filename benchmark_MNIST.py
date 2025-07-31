from kan import *
import numpy as np
import pandas as pd 
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os
import random


# -----------------------------------------------------------------------------
# LeNet-5 Implementation
# -----------------------------------------------------------------------------

class LeNet5(nn.Module):
    """LeNet-5 architecture for MNIST classification."""
    
    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)  # 28x28 -> 28x28
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, padding=0)  # 28x28 -> 24x24
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5, padding=0)  # 24x24 -> 20x20
        
        # Pooling layers
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # 28x28 -> 14x14 -> 12x12 -> 10x10
        
        # Fully connected layers
        self.fc1 = nn.Linear(120, 84)  # After pooling: 10x10 -> 5x5
        self.fc2 = nn.Linear(84, num_classes)
        
        # Activation functions
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        # x shape: (batch_size, 1, 28, 28)
        
        # Convolutional layers
        x = self.pool(self.relu(self.conv1(x)))  # -> 14x14
        x = self.pool(self.relu(self.conv2(x)))  # -> 12x12 -> 6x6
        x = self.relu(self.conv3(x))  # -> 2x2
        
        # Flatten
        x = x.view(x.size(0), -1)  # -> 120 * 5 * 5 = 3000
        
        # Fully connected layers
        x = self.tanh(self.fc1(x))
        x = self.fc2(x)
        
        return x


# -----------------------------------------------------------------------------
# Data Loading and Preparation
# -----------------------------------------------------------------------------

def load_mnist_data(train_samples=50000, test_samples=10000, device='cpu'):
    """
    Load MNIST dataset and prepare for KAN training.
    
    Args:
        train_samples: Number of training samples to use
        test_samples: Number of test samples to use
        device: Device to load data on
    
    Returns:
        dataset: Dictionary with train/test inputs and labels
    """
    print("Loading MNIST dataset...")
    
    # Load MNIST
    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST(root='../data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='../data', train=False, download=True, transform=transform)
    
    # Sample data
    if train_samples < len(train_dataset):
        train_indices = random.sample(range(len(train_dataset)), train_samples)
        train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
    
    if test_samples < len(test_dataset):
        test_indices = random.sample(range(len(test_dataset)), test_samples)
        test_dataset = torch.utils.data.Subset(test_dataset, test_indices)
    
    # Convert to tensors
    train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
    
    train_data, train_labels = next(iter(train_loader))
    test_data, test_labels = next(iter(test_loader))
    
    # Flatten images: (N, 1, 28, 28) -> (N, 784)
    train_input = train_data.view(train_data.size(0), -1).to(device)
    test_input = test_data.view(test_data.size(0), -1).to(device)
    
    # Convert labels to one-hot encoding for KAN
    train_label = torch.zeros(train_labels.size(0), 10, device=device)
    train_label.scatter_(1, train_labels.unsqueeze(1).to(device), 1)
    
    test_label = torch.zeros(test_labels.size(0), 10, device=device)
    test_label.scatter_(1, test_labels.unsqueeze(1).to(device), 1)
    
    dataset = {
        'train_input': train_input,
        'test_input': test_input,
        'train_label': train_label,
        'test_label': test_label,
        'train_labels_original': train_labels.to(device),
        'test_labels_original': test_labels.to(device)
    }
    
    print(f"Dataset prepared:")
    print(f"  Train samples: {train_input.shape[0]}")
    print(f"  Test samples: {test_input.shape[0]}")
    print(f"  Input dimensions: {train_input.shape[1]}")
    print(f"  Output dimensions: {train_label.shape[1]}")
    print(f"  Device: {device}")
    
    return dataset


def train_lenet5(dataset, epochs=10, device='cpu'):
    """
    Train LeNet-5 model on MNIST.
    
    Args:
        dataset: MNIST dataset
        epochs: Number of training epochs
        device: Device to train on
    
    Returns:
        model: Trained LeNet-5 model
    """
    print(f"Training LeNet-5 for {epochs} epochs on {device}...")
    
    model = LeNet5().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Prepare data for LeNet-5 (reshape back to images)
    train_input = dataset['train_input'].view(-1, 1, 28, 28).to(device)
    train_labels = dataset['train_labels_original'].to(device)
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(train_input)
        loss = criterion(outputs, train_labels)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
    
    return model


def evaluate_model(model, dataset, device='cpu'):
    """
    Evaluate model accuracy on test set.
    
    Args:
        model: Trained model
        dataset: Test dataset
        device: Device to evaluate on
    
    Returns:
        accuracy: Classification accuracy
    """
    model.eval()
    with torch.no_grad():
        test_input = dataset['test_input'].view(-1, 1, 28, 28).to(device)
        test_labels = dataset['test_labels_original'].to(device)
        
        outputs = model(test_input)
        _, predicted = torch.max(outputs.data, 1)
        accuracy = (predicted == test_labels).sum().item() / test_labels.size(0)
    
    return accuracy


# -----------------------------------------------------------------------------
# KAN Training Functions
# -----------------------------------------------------------------------------

def get_kan_structure(model):
    """
    Get the current structure of a KAN model as a list of layer widths.
    
    Args:
        model: KAN model
    
    Returns:
        structure: List representing the width of each layer
    """
    structure = []
    for layer in model.blocks:
        structure.append(layer.width)
    return structure


def train_kan_direct(dataset, structure, device='cpu', batch_size=256, prune=False, regularization=None):
    """
    Train KAN directly on MNIST data using batch processing.
    
    Args:
        dataset: MNIST dataset
        structure: KAN architecture [input_dim, hidden1, hidden2, ..., output_dim]
        device: Device to train on
        batch_size: Batch size for training to handle GPU memory
        prune: Whether to apply pruning after training and retrain
        regularization: Dictionary with regularization parameters
            - lamb: Overall penalty strength
            - lamb_l1: L1 penalty strength  
            - lamb_entropy: Entropy penalty strength
            - lamb_coef: Coefficient penalty strength
            - lamb_coefdiff: Coefficient smoothness penalty strength
            - reg_metric: Regularization metric ('edge_forward_spline_n', 'edge_forward_spline_u', 'edge_forward_sum', 'edge_backward', 'node_backward')
    
    Returns:
        model: Trained KAN model
        training_time: Time taken for training
        pruned_structure: Structure after pruning (None if not pruned)
    """
    print(f"Training KAN with structure: {structure} on {device} with batch_size={batch_size}")
    if prune:
        print("Pruning will be applied after training, followed by retraining")
    if regularization:
        print(f"Regularization enabled: {regularization}")
    
    start_time = time.time()
    pruned_structure = None
    
    # Create KAN model
    model = KAN(width=structure, grid=5, k=7, seed=42, device=device)
    
    # Create batch dataset for training
    train_input = dataset['train_input'].to(device)
    train_label = dataset['train_label'].to(device)
    
    # Split into batches
    num_samples = train_input.size(0)
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    print(f"Training with {num_batches} batches of size {batch_size}")
    
    # Set regularization parameters
    if regularization:
        lamb = regularization.get('lamb', 0.01)
        lamb_l1 = regularization.get('lamb_l1', 1.0)
        lamb_entropy = regularization.get('lamb_entropy', 2.0)
        lamb_coef = regularization.get('lamb_coef', 0.0)
        lamb_coefdiff = regularization.get('lamb_coefdiff', 0.0)
        reg_metric = regularization.get('reg_metric', 'edge_forward_spline_n')
    else:
        lamb = 0.0
        lamb_l1 = 1.0
        lamb_entropy = 2.0
        lamb_coef = 0.0
        lamb_coefdiff = 0.0
        reg_metric = 'edge_forward_spline_n'
    
    # Train KAN with batches
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, num_samples)
        
        batch_dataset = {
            'train_input': train_input[start_idx:end_idx],
            'train_label': train_label[start_idx:end_idx],
            'test_input': dataset['test_input'].to(device),
            'test_label': dataset['test_label'].to(device)
        }
        
        print(f"Training batch {batch_idx + 1}/{num_batches} (samples {start_idx}-{end_idx})")
        
        # Train on this batch with regularization
        if batch_idx == 0:
            # First batch: full training
            loss_result = model.fit(
                batch_dataset, 
                opt="LBFGS", 
                steps=20,
                lamb=lamb,
                lamb_l1=lamb_l1,
                lamb_entropy=lamb_entropy,
                lamb_coef=lamb_coef,
                lamb_coefdiff=lamb_coefdiff,
                reg_metric=reg_metric
            )
        else:
            # Subsequent batches: continue training
            loss_result = model.fit(
                batch_dataset, 
                opt="LBFGS", 
                steps=10,
                lamb=lamb,
                lamb_l1=lamb_l1,
                lamb_entropy=lamb_entropy,
                lamb_coef=lamb_coef,
                lamb_coefdiff=lamb_coefdiff,
                reg_metric=reg_metric
            )
    
    # Apply pruning if requested
    if prune:
        print("Applying pruning to the trained model...")
        # Get activations for pruning
        model(dataset['train_input'][:batch_size].to(device))
        # Apply pruning with default thresholds
        model = model.prune(node_th=1e-2, edge_th=3e-2)
        # Get the pruned structure
        pruned_structure = get_kan_structure(model)
        print(f"Pruning completed. New structure: {pruned_structure}")
        
        # Retrain the pruned model
        print("Retraining the pruned model...")
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_samples)
            
            batch_dataset = {
                'train_input': train_input[start_idx:end_idx],
                'train_label': train_label[start_idx:end_idx],
                'test_input': dataset['test_input'].to(device),
                'test_label': dataset['test_label'].to(device)
            }
            
            print(f"Retraining batch {batch_idx + 1}/{num_batches} (samples {start_idx}-{end_idx})")
            
            # Retrain on this batch with regularization
            if batch_idx == 0:
                # First batch: full retraining
                loss_result = model.fit(
                    batch_dataset, 
                    opt="LBFGS", 
                    steps=20,
                    lamb=lamb,
                    lamb_l1=lamb_l1,
                    lamb_entropy=lamb_entropy,
                    lamb_coef=lamb_coef,
                    lamb_coefdiff=lamb_coefdiff,
                    reg_metric=reg_metric
                )
            else:
                # Subsequent batches: continue retraining
                loss_result = model.fit(
                    batch_dataset, 
                    opt="LBFGS", 
                    steps=10,
                    lamb=lamb,
                    lamb_l1=lamb_l1,
                    lamb_entropy=lamb_entropy,
                    lamb_coef=lamb_coef,
                    lamb_coefdiff=lamb_coefdiff,
                    reg_metric=reg_metric
                )
        print("Retraining completed")
    
    training_time = time.time() - start_time
    
    return model, training_time, pruned_structure


def create_approximation_dataset(lenet_model, dataset, sample_points=1000, device='cpu', batch_size=256):
    """
    Create dataset for KAN approximation of LeNet-5 using batch processing.
    
    Args:
        lenet_model: Trained LeNet-5 model
        dataset: Original MNIST dataset
        sample_points: Number of points to sample
        device: Device to use
        batch_size: Batch size for processing
    
    Returns:
        approx_dataset: Dataset for KAN approximation
    """
    print(f"Creating approximation dataset with {sample_points} sample points on {device} using batch_size={batch_size}...")
    
    lenet_model.eval()
    
    # Randomly sample points from the training set
    total_train = dataset['train_input'].size(0)
    if sample_points > total_train:
        sample_points = total_train
    
    indices = random.sample(range(total_train), sample_points)
    
    # Get sampled inputs
    sampled_inputs = dataset['train_input'][indices].to(device)
    
    # Get LeNet-5 outputs for these inputs using batches
    lenet_outputs = []
    num_batches = (sample_points + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, sample_points)
            
            batch_inputs = sampled_inputs[start_idx:end_idx]
            lenet_inputs = batch_inputs.view(-1, 1, 28, 28).to(device)
            batch_outputs = lenet_model(lenet_inputs)
            lenet_outputs.append(batch_outputs)
    
    # Concatenate all batch outputs
    lenet_outputs = torch.cat(lenet_outputs, dim=0)
    
    # Create approximation dataset
    approx_dataset = {
        'train_input': sampled_inputs,
        'train_label': lenet_outputs,
        'test_input': dataset['test_input'].to(device),
        'test_label': dataset['test_label'].to(device)
    }
    
    return approx_dataset


def train_kan_approximation(dataset, structure, device='cpu', batch_size=256, prune=False, regularization=None):
    """
    Train KAN to approximate LeNet-5 using batch processing.
    
    Args:
        dataset: Approximation dataset
        structure: KAN architecture
        device: Device to train on
        batch_size: Batch size for training to handle GPU memory
        prune: Whether to apply pruning after training and retrain
        regularization: Dictionary with regularization parameters
            - lamb: Overall penalty strength
            - lamb_l1: L1 penalty strength  
            - lamb_entropy: Entropy penalty strength
            - lamb_coef: Coefficient penalty strength
            - lamb_coefdiff: Coefficient smoothness penalty strength
            - reg_metric: Regularization metric ('edge_forward_spline_n', 'edge_forward_spline_u', 'edge_forward_sum', 'edge_backward', 'node_backward')
    
    Returns:
        model: Trained KAN model
        training_time: Time taken for training
        pruned_structure: Structure after pruning (None if not pruned)
    """
    print(f"Training KAN approximation with structure: {structure} on {device} with batch_size={batch_size}")
    if prune:
        print("Pruning will be applied after training, followed by retraining")
    if regularization:
        print(f"Regularization enabled: {regularization}")
    
    start_time = time.time()
    pruned_structure = None
    
    # Create KAN model
    model = KAN(width=structure, grid=5, k=7, seed=42, device=device)
    
    # Create batch dataset for training
    train_input = dataset['train_input'].to(device)
    train_label = dataset['train_label'].to(device)
    
    # Split into batches
    num_samples = train_input.size(0)
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    print(f"Training with {num_batches} batches of size {batch_size}")
    
    # Set regularization parameters
    if regularization:
        lamb = regularization.get('lamb', 0.01)
        lamb_l1 = regularization.get('lamb_l1', 1.0)
        lamb_entropy = regularization.get('lamb_entropy', 2.0)
        lamb_coef = regularization.get('lamb_coef', 0.0)
        lamb_coefdiff = regularization.get('lamb_coefdiff', 0.0)
        reg_metric = regularization.get('reg_metric', 'edge_forward_spline_n')
    else:
        lamb = 0.0
        lamb_l1 = 1.0
        lamb_entropy = 2.0
        lamb_coef = 0.0
        lamb_coefdiff = 0.0
        reg_metric = 'edge_forward_spline_n'
    
    # Train KAN with batches
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, num_samples)
        
        batch_dataset = {
            'train_input': train_input[start_idx:end_idx],
            'train_label': train_label[start_idx:end_idx],
            'test_input': dataset['test_input'].to(device),
            'test_label': dataset['test_label'].to(device)
        }
        
        print(f"Training batch {batch_idx + 1}/{num_batches} (samples {start_idx}-{end_idx})")
        
        # Train on this batch with regularization
        if batch_idx == 0:
            # First batch: full training
            loss_result = model.fit(
                batch_dataset, 
                opt="LBFGS", 
                steps=20,
                lamb=lamb,
                lamb_l1=lamb_l1,
                lamb_entropy=lamb_entropy,
                lamb_coef=lamb_coef,
                lamb_coefdiff=lamb_coefdiff,
                reg_metric=reg_metric
            )
        else:
            # Subsequent batches: continue training
            loss_result = model.fit(
                batch_dataset, 
                opt="LBFGS", 
                steps=10,
                lamb=lamb,
                lamb_l1=lamb_l1,
                lamb_entropy=lamb_entropy,
                lamb_coef=lamb_coef,
                lamb_coefdiff=lamb_coefdiff,
                reg_metric=reg_metric
            )
    
    # Apply pruning if requested
    if prune:
        print("Applying pruning to the trained model...")
        # Get activations for pruning
        model(dataset['train_input'][:batch_size].to(device))
        # Apply pruning with default thresholds
        model = model.prune(node_th=1e-2, edge_th=3e-2)
        # Get the pruned structure
        pruned_structure = get_kan_structure(model)
        print(f"Pruning completed. New structure: {pruned_structure}")
        
        # Retrain the pruned model
        print("Retraining the pruned model...")
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_samples)
            
            batch_dataset = {
                'train_input': train_input[start_idx:end_idx],
                'train_label': train_label[start_idx:end_idx],
                'test_input': dataset['test_input'].to(device),
                'test_label': dataset['test_label'].to(device)
            }
            
            print(f"Retraining batch {batch_idx + 1}/{num_batches} (samples {start_idx}-{end_idx})")
            
            # Retrain on this batch with regularization
            if batch_idx == 0:
                # First batch: full retraining
                loss_result = model.fit(
                    batch_dataset, 
                    opt="LBFGS", 
                    steps=20,
                    lamb=lamb,
                    lamb_l1=lamb_l1,
                    lamb_entropy=lamb_entropy,
                    lamb_coef=lamb_coef,
                    lamb_coefdiff=lamb_coefdiff,
                    reg_metric=reg_metric
                )
            else:
                # Subsequent batches: continue retraining
                loss_result = model.fit(
                    batch_dataset, 
                    opt="LBFGS", 
                    steps=10,
                    lamb=lamb,
                    lamb_l1=lamb_l1,
                    lamb_entropy=lamb_entropy,
                    lamb_coef=lamb_coef,
                    lamb_coefdiff=lamb_coefdiff,
                    reg_metric=reg_metric
                )
        print("Retraining completed")
    
    training_time = time.time() - start_time
    
    return model, training_time, pruned_structure


# -----------------------------------------------------------------------------
# Evaluation Functions
# -----------------------------------------------------------------------------

def evaluate_kan_classification(model, dataset, device='cpu'):
    """
    Evaluate KAN model for classification accuracy.
    
    Args:
        model: Trained KAN model
        dataset: Test dataset
        device: Device to evaluate on
    
    Returns:
        accuracy: Classification accuracy
    """
    model.eval()
    with torch.no_grad():
        test_input = dataset['test_input'].to(device)
        test_label = dataset['test_labels_original'].to(device)
        
        # Get KAN predictions
        outputs = model(test_input)
        
        # For direct KAN training, outputs are logits
        if outputs.shape[1] == 10:  # 10 classes
            _, predicted = torch.max(outputs.data, 1)
            accuracy = (predicted == test_label).sum().item() / test_label.size(0)
        else:
            # For single output, round to nearest class
            predicted = torch.round(outputs).long().squeeze()
            accuracy = (predicted == test_label).sum().item() / test_label.size(0)
    
    return accuracy


def evaluate_kan_approximation(model, dataset, lenet_model, device='cpu'):
    """
    Evaluate KAN approximation of LeNet-5.
    
    Args:
        model: Trained KAN model
        dataset: Test dataset
        lenet_model: Original LeNet-5 model
        device: Device to evaluate on
    
    Returns:
        mse_error: Mean squared error
        max_error: Maximum absolute error
        mean_error: Mean absolute error
    """
    model.eval()
    lenet_model.eval()
    
    with torch.no_grad():
        test_input = dataset['test_input'].to(device)
        
        # Get LeNet-5 outputs
        lenet_input = test_input.reshape(-1, 1, 28, 28).to(device)
        lenet_outputs = lenet_model(lenet_input)
        
        # Get KAN outputs
        kan_outputs = model(test_input)
        
        # Calculate errors
        mse_error = torch.mean((kan_outputs - lenet_outputs) ** 2)
        abs_error = torch.abs(kan_outputs - lenet_outputs)
        max_error = torch.max(abs_error)
        mean_error = torch.mean(abs_error)
    
    return mse_error.item(), max_error.item(), mean_error.item()


def model_evaluate(model, data, zero_mask=1e-3):
    """
    Evaluate the model on the dataset (adapted from benchmark.py).
    
    Args:
        model: Trained model
        data: Test dataset
        zero_mask: Threshold for relative error calculation
    
    Returns:
        error_dict: Dictionary with error metrics
    """
    model.eval()
    with torch.no_grad():
        model_predict = model.forward(data["test_input"])
        
        mse_error = torch.sqrt(torch.mean((model_predict - data["test_label"])**2))
        abs_error = torch.sqrt((model_predict - data["test_label"])**2)
        relative_error = torch.where(
            torch.abs(data["test_label"]) >= zero_mask,
            abs_error / torch.abs(data["test_label"]),
            abs_error
        )
        mean_relative_error = torch.mean(relative_error)
        maximum_relative_error = torch.max(relative_error)
        
        error_dict = {
            "MSE": mse_error.item(),
            "Mean Relative Error": mean_relative_error.item(),
            "Maximum Relative Error": maximum_relative_error.item()
        }
        
        print("Evaluation Results:")
        for key, value in error_dict.items():
            print(f"{key}: {value:.6f}")
            
        return error_dict


# -----------------------------------------------------------------------------
# Main Benchmark Functions
# -----------------------------------------------------------------------------

def run_direct_kan_benchmark(dataset, kan_structures, device='cpu', batch_size=256, prune=False, regularization=None):
    """
    Run benchmark for direct KAN training on MNIST.
    
    Args:
        dataset: MNIST dataset
        kan_structures: List of KAN architectures to test
        device: Device to use
        batch_size: Batch size for training
        prune: Whether to apply pruning after training
        regularization: Dictionary with regularization parameters
    
    Returns:
        results: List of results for each structure
    """
    print("\n" + "="*50)
    print("DIRECT KAN TRAINING BENCHMARK")
    print("="*50)
    if prune:
        print("Pruning enabled")
    if regularization:
        print(f"Regularization enabled: {regularization}")
    
    results = []
    
    for i, structure in enumerate(kan_structures):
        print(f"\nTesting KAN structure {i+1}/{len(kan_structures)}: {structure}")
        
        try:
            # Train KAN directly
            model, training_time, pruned_structure = train_kan_direct(dataset, structure, device, batch_size, prune, regularization)
            
            # Evaluate classification accuracy
            accuracy = evaluate_kan_classification(model, dataset, device)
            
            # Evaluate regression metrics
            error_dict = model_evaluate(model, dataset)
            
            # Record results for original structure
            results.append({
                "Method": "Direct KAN",
                "Structure": str(structure),
                "Training Time (s)": training_time,
                "Classification Accuracy": accuracy,
                "MSE": error_dict["MSE"],
                "Mean Relative Error": error_dict["Mean Relative Error"],
                "Maximum Relative Error": error_dict["Maximum Relative Error"],
                "Batch Size": batch_size,
                "Pruned": prune,
                "Regularization": str(regularization) if regularization else "None",
                "Original Structure": str(structure),
                "Pruned Structure": str(pruned_structure) if prune else "N/A"
            })
            
            print(f"Direct KAN Results:")
            print(f"  Training Time: {training_time:.2f}s")
            print(f"  Classification Accuracy: {accuracy:.4f}")
            print(f"  MSE: {error_dict['MSE']:.6f}")
            
            # If pruning was applied, add an additional row for the pruned structure
            if prune and pruned_structure is not None:
                print(f"  Pruned Structure: {pruned_structure}")
                # Add a separate row for the pruned structure
                results.append({
                    "Method": "Direct KAN (Pruned)",
                    "Structure": str(pruned_structure),
                    "Training Time (s)": training_time,  # Same training time since it's the same model
                    "Classification Accuracy": accuracy,  # Same accuracy since it's the same model
                    "MSE": error_dict["MSE"],  # Same MSE since it's the same model
                    "Mean Relative Error": error_dict["Mean Relative Error"],
                    "Maximum Relative Error": error_dict["Maximum Relative Error"],
                    "Batch Size": batch_size,
                    "Pruned": True,
                    "Regularization": str(regularization) if regularization else "None",
                    "Original Structure": str(structure),
                    "Pruned Structure": str(pruned_structure)
                })
            
        except Exception as e:
            print(f"Error with structure {structure}: {e}")
            results.append({
                "Method": "Direct KAN",
                "Structure": str(structure),
                "Training Time (s)": -1,
                "Classification Accuracy": -1,
                "MSE": -1,
                "Mean Relative Error": -1,
                "Maximum Relative Error": -1,
                "Batch Size": batch_size,
                "Pruned": prune,
                "Regularization": str(regularization) if regularization else "None",
                "Original Structure": str(structure),
                "Pruned Structure": "N/A",
                "Error": str(e)
            })
    
    return results


def run_approximation_kan_benchmark(dataset, lenet_model, kan_structures, sample_points=1000, device='cpu', batch_size=256, prune=False, regularization=None):
    """
    Run benchmark for KAN approximation of LeNet-5.
    
    Args:
        dataset: MNIST dataset
        lenet_model: Trained LeNet-5 model
        kan_structures: List of KAN architectures to test
        sample_points: Number of points to sample for approximation
        device: Device to use
        batch_size: Batch size for training
        prune: Whether to apply pruning after training
        regularization: Dictionary with regularization parameters
    
    Returns:
        results: List of results for each structure
    """
    print("\n" + "="*50)
    print("KAN APPROXIMATION OF LENET-5 BENCHMARK")
    print("="*50)
    if prune:
        print("Pruning enabled")
    if regularization:
        print(f"Regularization enabled: {regularization}")
    
    # Create approximation dataset
    approx_dataset = create_approximation_dataset(lenet_model, dataset, sample_points, device, batch_size)
    
    results = []
    
    for i, structure in enumerate(kan_structures):
        print(f"\nTesting KAN structure {i+1}/{len(kan_structures)}: {structure}")
        
        try:
            # Train KAN for approximation
            model, training_time, pruned_structure = train_kan_approximation(approx_dataset, structure, device, batch_size, prune, regularization)
            
            # Evaluate approximation quality
            mse_error, max_error, mean_error = evaluate_kan_approximation(
                model, dataset, lenet_model, device
            )
            
            # Evaluate classification accuracy of approximated model
            accuracy = evaluate_kan_classification(model, dataset, device)
            
            # Record results for original structure
            results.append({
                "Method": "KAN Approximation",
                "Structure": str(structure),
                "Sample Points": sample_points,
                "Training Time (s)": training_time,
                "Classification Accuracy": accuracy,
                "MSE": mse_error,
                "Mean Absolute Error": mean_error,
                "Maximum Absolute Error": max_error,
                "Batch Size": batch_size,
                "Pruned": prune,
                "Regularization": str(regularization) if regularization else "None",
                "Original Structure": str(structure),
                "Pruned Structure": str(pruned_structure) if prune else "N/A"
            })
            
            print(f"KAN Approximation Results:")
            print(f"  Training Time: {training_time:.2f}s")
            print(f"  Classification Accuracy: {accuracy:.4f}")
            print(f"  MSE: {mse_error:.6f}")
            print(f"  Mean Absolute Error: {mean_error:.6f}")
            print(f"  Maximum Absolute Error: {max_error:.6f}")
            
            # If pruning was applied, add an additional row for the pruned structure
            if prune and pruned_structure is not None:
                print(f"  Pruned Structure: {pruned_structure}")
                # Add a separate row for the pruned structure
                results.append({
                    "Method": "KAN Approximation (Pruned)",
                    "Structure": str(pruned_structure),
                    "Sample Points": sample_points,
                    "Training Time (s)": training_time,  # Same training time since it's the same model
                    "Classification Accuracy": accuracy,  # Same accuracy since it's the same model
                    "MSE": mse_error,  # Same MSE since it's the same model
                    "Mean Absolute Error": mean_error,
                    "Maximum Absolute Error": max_error,
                    "Batch Size": batch_size,
                    "Pruned": True,
                    "Regularization": str(regularization) if regularization else "None",
                    "Original Structure": str(structure),
                    "Pruned Structure": str(pruned_structure)
                })
            
        except Exception as e:
            print(f"Error with structure {structure}: {e}")
            results.append({
                "Method": "KAN Approximation",
                "Structure": str(structure),
                "Sample Points": sample_points,
                "Training Time (s)": -1,
                "Classification Accuracy": -1,
                "MSE": -1,
                "Mean Absolute Error": -1,
                "Maximum Absolute Error": -1,
                "Batch Size": batch_size,
                "Pruned": prune,
                "Regularization": str(regularization) if regularization else "None",
                "Original Structure": str(structure),
                "Pruned Structure": "N/A",
                "Error": str(e)
            })
    
    return results


# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------

if __name__ == "__main__":

    os.environ['HTTP_PROXY']="http://Clash:QOAF8Rmd@10.1.0.213:7890" 
    os.environ['HTTPS_PROXY']="http://Clash:QOAF8Rmd@10.1.0.213:7890" 
    os.environ['ALL_PROXY']="socks5://Clash:QOAF8Rmd@10.1.0.213:7893"
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Device configuration
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cuda:3')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Configuration parameters
    train_samples = 50000  # Number of training samples to use
    test_samples = 10000    # Number of test samples to use
    lenet_epochs = 300      # Number of epochs for LeNet-5 training
    sample_points_list = [100000, 200000, 500000]  # Different numbers of sample points for approximation
    batch_size = 4096        # Batch size for KAN training to handle GPU memory
    
    # KAN structures to test (input_dim -> hidden_layers -> output_dim)
    # For MNIST: input_dim = 784 (28*28), output_dim = 10 (classes)
    kan_structures = [
        [784, 50, 20, 10],
        [784, 30, 30, 20, 10],    
        [784, 20, 20, 20, 10, 10, 10],
    ]
    
    # Regularization configurations to test
    regularization_configs = [
        None  # No regularization
        # {
        #     'lamb': 0.01,
        #     'lamb_l1': 1.0,
        #     'lamb_entropy': 2.0,
        #     'lamb_coef': 0.0,
        #     'lamb_coefdiff': 0.0,
        #     'reg_metric': 'edge_forward_spline_n'
        # },
        # {
        #     'lamb': 0.05,
        #     'lamb_l1': 1.0,
        #     'lamb_entropy': 2.0,
        #     'lamb_coef': 0.1,
        #     'lamb_coefdiff': 0.1,
        #     'reg_metric': 'edge_forward_spline_n'
        # }
    ]
    
    # Pruning configurations to test
    pruning_configs = [False]
    
    # Load MNIST dataset
    print("Loading MNIST dataset...")
    dataset = load_mnist_data(train_samples, test_samples, device)
    
    # Train LeNet-5 for comparison
    print("\nTraining LeNet-5 for comparison...")
    lenet_model = train_lenet5(dataset, lenet_epochs, device=device)
    lenet_accuracy = evaluate_model(lenet_model, dataset, device)
    print(f"LeNet-5 Test Accuracy: {lenet_accuracy:.4f}")
    
    # Run benchmarks
    all_results = []
    
    # Test different combinations of pruning and regularization
    for prune in pruning_configs:
        for regularization in regularization_configs:
            print(f"\n{'='*60}")
            print(f"Testing with prune={prune}, regularization={regularization}")
            print(f"{'='*60}")
            
            # 1. Direct KAN training benchmark
            direct_results = run_direct_kan_benchmark(
                dataset, kan_structures, device, batch_size, prune, regularization
            )
            all_results.extend(direct_results)
            
            # 2. KAN approximation benchmark with different sample points
            for sample_points in sample_points_list:
                print(f"\nRunning approximation benchmark with {sample_points} sample points...")
                approx_results = run_approximation_kan_benchmark(
                    dataset, lenet_model, kan_structures, sample_points, device, batch_size, prune, regularization
                )
                all_results.extend(approx_results)
    
    # Save results
    results_df = pd.DataFrame(all_results)
    
    # Create results directory
    results_dir = f"MNIST_results_{device}"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # Save to Excel file
    results_file = os.path.join(results_dir, "benchmark_results.xlsx")
    results_df.to_excel(results_file, index=False)
    print(f"\nResults saved to: {results_file}")
    
    # Print summary
    print("\n" + "="*50)
    print("BENCHMARK SUMMARY")
    print("="*50)
    print(f"LeNet-5 Baseline Accuracy: {lenet_accuracy:.4f}")
    print(f"Total experiments: {len(all_results)}")
    print(f"Results saved to: {results_file}")
    
    # Print best results for each method and configuration
    print("\nBest Direct KAN Results by Configuration:")
    direct_df = results_df[results_df['Method'] == 'Direct KAN']
    direct_pruned_df = results_df[results_df['Method'] == 'Direct KAN (Pruned)']
    if not direct_df.empty:
        for prune in pruning_configs:
            for reg in regularization_configs:
                reg_str = str(reg) if reg else "None"
                subset = direct_df[(direct_df['Pruned'] == prune) & (direct_df['Regularization'] == reg_str)]
                if not subset.empty:
                    best_direct = subset.loc[subset['Classification Accuracy'].idxmax()]
                    print(f"  Prune={prune}, Reg={reg_str}:")
                    print(f"    Best Accuracy: {best_direct['Classification Accuracy']:.4f}")
                    print(f"    Structure: {best_direct['Structure']}")
                    print(f"    Training Time: {best_direct['Training Time (s)']:.2f}s")
                    if prune and not direct_pruned_df.empty:
                        pruned_subset = direct_pruned_df[(direct_pruned_df['Pruned'] == True) & (direct_pruned_df['Regularization'] == reg_str)]
                        if not pruned_subset.empty:
                            best_pruned = pruned_subset.loc[pruned_subset['Classification Accuracy'].idxmax()]
                            print(f"    Pruned Structure: {best_pruned['Structure']}")
                            print(f"    Pruned Accuracy: {best_pruned['Classification Accuracy']:.4f}")
    
    print("\nBest KAN Approximation Results by Configuration:")
    approx_df = results_df[results_df['Method'] == 'KAN Approximation']
    approx_pruned_df = results_df[results_df['Method'] == 'KAN Approximation (Pruned)']
    if not approx_df.empty:
        for prune in pruning_configs:
            for reg in regularization_configs:
                reg_str = str(reg) if reg else "None"
                subset = approx_df[(approx_df['Pruned'] == prune) & (approx_df['Regularization'] == reg_str)]
                if not subset.empty:
                    best_approx = subset.loc[subset['Classification Accuracy'].idxmax()]
                    print(f"  Prune={prune}, Reg={reg_str}:")
                    print(f"    Best Accuracy: {best_approx['Classification Accuracy']:.4f}")
                    print(f"    Structure: {best_approx['Structure']}")
                    print(f"    Sample Points: {best_approx['Sample Points']}")
                    print(f"    Training Time: {best_approx['Training Time (s)']:.2f}s")
                    print(f"    MSE: {best_approx['MSE']:.6f}")
                    if prune and not approx_pruned_df.empty:
                        pruned_subset = approx_pruned_df[(approx_pruned_df['Pruned'] == True) & (approx_pruned_df['Regularization'] == reg_str)]
                        if not pruned_subset.empty:
                            best_pruned = pruned_subset.loc[pruned_subset['Classification Accuracy'].idxmax()]
                            print(f"    Pruned Structure: {best_pruned['Structure']}")
                            print(f"    Pruned Accuracy: {best_pruned['Classification Accuracy']:.4f}")
                            print(f"    Pruned MSE: {best_pruned['MSE']:.6f}")
    
    # Print overall best results
    print("\nOverall Best Results:")
    if not direct_df.empty:
        overall_best_direct = direct_df.loc[direct_df['Classification Accuracy'].idxmax()]
        print(f"  Best Direct KAN: {overall_best_direct['Classification Accuracy']:.4f} accuracy")
        print(f"    Configuration: Prune={overall_best_direct['Pruned']}, Reg={overall_best_direct['Regularization']}")
        print(f"    Structure: {overall_best_direct['Structure']}")
        if overall_best_direct['Pruned'] and not direct_pruned_df.empty:
            # Find corresponding pruned result
            pruned_match = direct_pruned_df[
                (direct_pruned_df['Original Structure'] == overall_best_direct['Structure']) &
                (direct_pruned_df['Regularization'] == overall_best_direct['Regularization'])
            ]
            if not pruned_match.empty:
                pruned_result = pruned_match.iloc[0]
                print(f"    Pruned Structure: {pruned_result['Structure']}")
    
    if not approx_df.empty:
        overall_best_approx = approx_df.loc[approx_df['Classification Accuracy'].idxmax()]
        print(f"  Best KAN Approximation: {overall_best_approx['Classification Accuracy']:.4f} accuracy")
        print(f"    Configuration: Prune={overall_best_approx['Pruned']}, Reg={overall_best_approx['Regularization']}")
        print(f"    Structure: {overall_best_approx['Structure']}")
        print(f"    Sample Points: {overall_best_approx['Sample Points']}")
        if overall_best_approx['Pruned'] and not approx_pruned_df.empty:
            # Find corresponding pruned result
            pruned_match = approx_pruned_df[
                (approx_pruned_df['Original Structure'] == overall_best_approx['Structure']) &
                (approx_pruned_df['Regularization'] == overall_best_approx['Regularization']) &
                (approx_pruned_df['Sample Points'] == overall_best_approx['Sample Points'])
            ]
            if not pruned_match.empty:
                pruned_result = pruned_match.iloc[0]
                print(f"    Pruned Structure: {pruned_result['Structure']}")
    
    # Print structure comparison summary
    print("\nStructure Comparison Summary:")
    if not direct_pruned_df.empty:
        print("Direct KAN Structure Changes:")
        for _, row in direct_pruned_df.iterrows():
            original = eval(row['Original Structure'])
            pruned = eval(row['Structure'])
            reduction = [(orig - pruned[i]) for i, orig in enumerate(original) if i < len(pruned)]
            print(f"  {original} -> {pruned} (Reduction: {reduction})")
    
    if not approx_pruned_df.empty:
        print("KAN Approximation Structure Changes:")
        for _, row in approx_pruned_df.iterrows():
            original = eval(row['Original Structure'])
            pruned = eval(row['Structure'])
            reduction = [(orig - pruned[i]) for i, orig in enumerate(original) if i < len(pruned)]
            print(f"  {original} -> {pruned} (Reduction: {reduction})")