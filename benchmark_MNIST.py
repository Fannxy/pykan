from kan import *
import numpy as np
import pandas as pd 
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os
import random
import math
import datetime
from tqdm import tqdm

# Define a global log filename
LOG_FILE = "cuda_memory_log.txt"

# If log file exists, delete it for fresh run
if os.path.exists(LOG_FILE):
    os.remove(LOG_FILE)

def log_cuda_memory(message):
    """
    Log current CUDA memory usage to a file.

    Args:
        message (str): Description of the current operation.
    """
    if not torch.cuda.is_available():
        print("CUDA is not available.")
        return

    allocated = torch.cuda.memory_allocated() / 1024**2  # MB
    reserved = torch.cuda.memory_reserved() / 1024**2   # MB
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
    
    log_message = (
        f"[{timestamp}] {message}\n"
        f"  - Allocated: {allocated:.2f} MB\n"
        f"  - Reserved:  {reserved:.2f} MB\n"
        f"--------------------------------------------------\n"
    )
    
    with open(LOG_FILE, "a") as f:
        f.write(log_message)


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
        train_dataset, test_dataset
    """
    print("Loading MNIST dataset...")
    
    # Load MNIST with normalization
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset_full = datasets.MNIST(root='../data', train=True, download=True, transform=transform)
    test_dataset_full = datasets.MNIST(root='../data', train=False, download=True, transform=transform)
    
    # Sample data
    train_indices = random.sample(range(len(train_dataset_full)), train_samples)
    test_indices = random.sample(range(len(test_dataset_full)), test_samples)
    
    train_dataset = torch.utils.data.Subset(train_dataset_full, train_indices)
    test_dataset = torch.utils.data.Subset(test_dataset_full, test_indices)
    
    print(f"Dataset prepared:")
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")
    print(f"  Device: {device}")
    
    return train_dataset, test_dataset


def train_lenet5(train_dataset, epochs=10, batch_size=128, device='cpu'):
    """
    Train LeNet-5 model on MNIST.
    
    Args:
        train_dataset: MNIST train dataset
        epochs: Number of training epochs
        batch_size: Batch size for training
        device: Device to train on
    
    Returns:
        model: Trained LeNet-5 model
    """
    print(f"Training LeNet-5 for {epochs} epochs on {device} with batch_size={batch_size}...")
    
    model = LeNet5().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}")
    
    return model


def evaluate_model(model, test_dataset, batch_size=256, device='cpu'):
    """
    Evaluate model accuracy on test set.
    
    Args:
        model: Trained model
        test_dataset: Test dataset
        batch_size: Batch size for evaluation
        device: Device to evaluate on
    
    Returns:
        accuracy: Classification accuracy
    """
    model.eval()
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / total
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

def train_kan_direct(train_dataset, test_dataset, structure, device='cpu', batch_size=256, prune=False, regularization=None):
    """
    Train KAN directly on MNIST data using batch processing.
    
    Args:
        train_dataset: MNIST train dataset
        test_dataset: MNIST test dataset
        structure: KAN architecture [input_dim, hidden1, hidden2, ..., output_dim]
        device: Device to train on
        batch_size: Batch size for training to handle GPU memory
        prune: Whether to apply pruning after training and retrain
        regularization: Dictionary with regularization parameters
    
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
    model = KAN(width=structure, grid=5, k=6, seed=42, device=device, auto_save=False)
    
    log_cuda_memory("Direct KAN training: After creating KAN model")
    
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
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    
    # Train KAN with batches
    for train_images, train_labels in train_loader:

        test_images, test_labels = next(iter(test_loader))
        test_input = test_images.view(test_images.size(0), -1).to(device)
        test_label = test_labels.to(device)

        train_input = train_images.view(train_images.size(0), -1).to(device)
        train_label = train_labels.to(device)
        batch_dataset = {
            'train_input': train_input,
            'train_label': train_label,
            'test_input': test_input,
            'test_label': test_label
        }
        
        log_cuda_memory("Direct KAN training: After loading batch data, before training batch")
        
        loss_result = model.fit(
            batch_dataset, 
            opt="LBFGS", 
            loss_fn=nn.CrossEntropyLoss(),
            steps=10,
            batch=batch_size,
            lamb=lamb,
            lamb_l1=lamb_l1,
            lamb_entropy=lamb_entropy,
            lamb_coef=lamb_coef,
            lamb_coefdiff=lamb_coefdiff,
            reg_metric=reg_metric
        )
        log_cuda_memory("Direct KAN training: After training batch")

    # Apply pruning if requested
    if prune:
        pass
    
    training_time = time.time() - start_time
    
    return model, training_time, pruned_structure

class GeneratedDataset(Dataset):
    """
    A dataset generated by teacher model labels.
    """
    def __init__(self, inputs, labels):
        assert len(inputs) == len(labels), "Inputs and labels must have the same length"
        self.inputs = inputs
        self.labels = labels

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]

def create_dataset_from_teacher(strategy, teacher_model, sample_points, batch_size, device, ori_dataset=None, chebyshev_degree=10):
    """
    Generate dataset using teacher model.
    
    Args:
        strategy (str): 'random', 'chebyshev', or 'ori_input'.
        teacher_model (nn.Module): Pretrained teacher model.
        sample_points (int): Total samples to generate.
        batch_size (int): Batch size for generation.
        device (torch.device): Device to run on.
        ori_dataset: Original dataset for 'ori_input'.
        chebyshev_degree: Degree for Chebyshev points.
    
    Returns:
        GeneratedDataset: Dataset with generated inputs and labels.
    """
    print(f"Generating {sample_points} samples using strategy '{strategy}'...")
    
    teacher_model.eval()
    
    all_inputs = []
    all_labels = []
    
    with torch.no_grad():
        ori_dataloader = None
        iterator = None
        if strategy == 'ori_input':
            ori_dataloader = DataLoader(ori_dataset, batch_size=batch_size, shuffle=False)
            iterator = iter(ori_dataloader)
        
        num_batches = (sample_points + batch_size - 1) // batch_size
        
        for _ in tqdm(range(num_batches)):
            if strategy == 'random':
                inputs = torch.rand(batch_size, 1, 28, 28, device=device)
            elif strategy == 'chebyshev':
                def generate_chebyshev_tensor(batch_size: int, n: int) -> torch.Tensor:
                    if n < 2:
                        raise ValueError("n must be >= 2")
                    indices = torch.randint(0, n, (batch_size, 1, 28, 28))
                    indices_float = indices.float()
                    chebyshev_points = (torch.cos(indices_float * math.pi / (n - 1)) + 1) / 2.0
                    return chebyshev_points
                inputs = generate_chebyshev_tensor(batch_size, chebyshev_degree).to(device)
            elif strategy == 'ori_input':
                try:
                    inputs, _ = next(iterator)
                    # if inputs.shape[0] < batch_size:
                    #     added_inputs = torch.rand(batch_size - inputs.shape[0], 1, 28, 28, device=device)
                    #     inputs = torch.cat([inputs.to(device), added_inputs], dim=0)
                    # else:
                    #     inputs = inputs.to(device)
                    inputs = inputs.to(device)
                except StopIteration:
                    # inputs = torch.rand(batch_size, 1, 28, 28, device=device)
                    break
            else:
                raise ValueError(f"Invalid strategy: {strategy}")
            
            labels = teacher_model(inputs)
            
            all_inputs.append(inputs.cpu())
            all_labels.append(labels.cpu())
    
    final_inputs = torch.cat(all_inputs, dim=0)[:sample_points]
    final_labels = torch.cat(all_labels, dim=0)[:sample_points]
    
    print(f"Generation complete! Dataset size: {final_inputs.shape}, {final_labels.shape}")
    
    return GeneratedDataset(final_inputs, final_labels)

def train_kan_approximation(train_dataset, test_dataset, structure, device='cpu', batch_size=256, prune=False, regularization=None):
    """
    Train KAN to approximate LeNet-5 using batch processing.
    
    Args:
        train_dataset: Approximation train dataset
        test_dataset: Approximation test dataset
        structure: KAN architecture
        device: Device to train on
        batch_size: Batch size for training
        prune: Whether to apply pruning
        regularization: Regularization parameters
    
    Returns:
        model, training_time, pruned_structure
    """
    print(f"Training KAN approximation with structure: {structure} on {device} with batch_size={batch_size}")
    if prune:
        print("Pruning will be applied after training, followed by retraining")
    if regularization:
        print(f"Regularization enabled: {regularization}")
    
    start_time = time.time()
    pruned_structure = None
    
    log_cuda_memory("Before creating KAN model")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    # Create KAN model
    model = KAN(width=structure, grid=10, k=6, seed=42, device=device, auto_save=False)
    
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
    
    log_cuda_memory("Before training")
    
    # Train KAN with batches
    for train_images, train_labels in train_loader:
        
        test_images, test_labels = next(iter(test_loader))
        test_input = test_images.view(test_images.size(0), -1).to(device)
        test_label = test_labels.to(device)

        train_input = train_images.view(train_images.size(0), -1).to(device)
        train_label = train_labels.to(device)

        batch_dataset = {
            'train_input': train_input,
            'train_label': train_label,
            'test_input': test_input,
            'test_label': test_label
        }
        
        log_cuda_memory("After loading batch data, before training batch")
        
        # Train on this batch with regularization
        loss_result = model.fit(
            batch_dataset, 
            opt="LBFGS", 
            steps=10,
            batch=batch_size,
            lamb=lamb,
            lamb_l1=lamb_l1,
            lamb_entropy=lamb_entropy,
            lamb_coef=lamb_coef,
            lamb_coefdiff=lamb_coefdiff,
            reg_metric=reg_metric
        )
        log_cuda_memory("After training batch")
        

    # Apply pruning if requested
    if prune:
        pass
    
    training_time = time.time() - start_time
    
    return model, training_time, pruned_structure


# -----------------------------------------------------------------------------
# Evaluation Functions
# -----------------------------------------------------------------------------

def evaluate_kan_classification(model, dataset, device='cpu', batch_size=256):
    """
    Evaluate KAN model for classification accuracy.
    
    Args:
        model: Trained KAN model
        dataset: Test dataset
        device: Device to evaluate on
        batch_size: Batch size
    
    Returns:
        accuracy: Classification accuracy
    """
    model.eval()
    with torch.no_grad():
        test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        correct = 0
        total = 0
        for inputs, labels in test_loader:
            inputs = inputs.view(inputs.size(0), -1).to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        accuracy = correct / total
    
    return accuracy

def evaluate_kan_approximation(model, approx_test_dataset, device='cpu', batch_size=256):
    """
    Evaluate KAN approximation of LeNet-5.
    
    Args:
        model: Trained KAN model
        approx_test_dataset: Approximation test dataset with teacher labels
        device: Device to evaluate on
        batch_size: Batch size
    
    Returns:
        mse_error, max_error, mean_error
    """
    model.eval()
    
    with torch.no_grad():
        test_loader = DataLoader(approx_test_dataset, batch_size=batch_size, shuffle=False)
        mse_error = 0
        max_error = []
        mean_error = 0
        for inputs, labels in test_loader:
            inputs = inputs.view(inputs.size(0), -1).to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            mse_error += torch.mean((outputs - labels) ** 2)
            max_error.append(torch.max(torch.abs(outputs - labels)))
            mean_error += torch.mean(torch.abs(outputs - labels))
        mse_error = mse_error / len(test_loader)
        max_error = max(max_error)
        mean_error = mean_error / len(test_loader)
    
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

def run_direct_kan_benchmark(train_dataset, test_dataset, kan_structures, device='cpu', batch_size=256, prune=False, regularization=None):
    """
    Run benchmark for direct KAN training on MNIST.
    
    Args:
        train_dataset, test_dataset: Datasets
        kan_structures: List of structures
        device: Device
        batch_size: Batch size
        prune: Prune flag
        regularization: Reg params
    
    Returns:
        results, models_list
    """
    print("\n" + "="*50)
    print("DIRECT KAN TRAINING BENCHMARK")
    print("="*50)
    if prune:
        print("Pruning enabled")
    if regularization:
        print(f"Regularization enabled: {regularization}")
    
    results = []
    models_list = []
    
    for i, structure in enumerate(kan_structures):
        print(f"\nTesting KAN structure {i+1}/{len(kan_structures)}: {structure}")
        
        try:
            log_cuda_memory("Before training KAN directly")
            # Train KAN directly
            model, training_time, pruned_structure = train_kan_direct(train_dataset, test_dataset, structure, device, batch_size, prune, regularization)
            
            log_cuda_memory("After training KAN directly, before evaluating classification accuracy")
            
            # Evaluate classification accuracy
            accuracy = evaluate_kan_classification(model, test_dataset, device, batch_size)
            
            log_cuda_memory("After evaluating classification accuracy in Direct KAN training")

            # Record results for original structure
            results.append({
                "Method": "Direct KAN",
                "Structure": str(structure),
                "Training Time (s)": training_time,
                "Classification Accuracy": accuracy,
                "MSE": "N/A",
                "Mean Relative Error": "N/A",
                "Maximum Relative Error": "N/A",
                "Batch Size": batch_size,
                "Pruned": prune,
                "Regularization": str(regularization) if regularization else "None",
                "Original Structure": str(structure),
                "Pruned Structure": str(pruned_structure) if prune else "N/A"
            })
            
            print(f"Direct KAN Results:")
            print(f"  Training Time: {training_time:.2f}s")
            print(f"  Classification Accuracy: {accuracy:.4f}")
            
            if prune and pruned_structure is not None:
                print(f"  Pruned Structure: {pruned_structure}")
                results.append({
                    "Method": "Direct KAN (Pruned)",
                    "Structure": str(pruned_structure),
                    "Training Time (s)": training_time,
                    "Classification Accuracy": accuracy,
                    "MSE": "N/A",
                    "Mean Relative Error": "N/A",
                    "Maximum Relative Error": "N/A",
                    "Batch Size": batch_size,
                    "Pruned": True,
                    "Regularization": str(regularization) if regularization else "None",
                    "Original Structure": str(structure),
                    "Pruned Structure": str(pruned_structure)
                })
            
            meta = {
                "method": "Direct KAN",
                "original_structure": str(structure),
                "pruned_structure": str(pruned_structure) if prune else "N/A",
                "prune": prune,
                "regularization": str(regularization) if regularization else "None",
                "accuracy": accuracy,
                "training_time": training_time,
                "batch_size": batch_size
            }
            model_cpu = model.to('cpu')
            models_list.append({"meta": meta, "model": model_cpu})
            del model
            torch.cuda.empty_cache()
            
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
    
    return results, models_list

def run_approximation_kan_benchmark(train_dataset, test_dataset, lenet_model, kan_structures, sample_points=(10000, 1000), device='cpu', batch_size=256, prune=False, regularization=None):
    """
    Run benchmark for KAN approximation of LeNet-5.
    
    Args:
        train_dataset, test_dataset: Original datasets
        lenet_model: Trained LeNet-5
        kan_structures: Structures
        sample_points: (train_samples, test_samples) for approximation
        device: Device
        batch_size: Batch size
        prune: Prune flag
        regularization: Reg params
    
    Returns:
        results, models_list
    """
    print("\n" + "="*50)
    print("KAN APPROXIMATION OF LENET-5 BENCHMARK")
    print("="*50)
    if prune:
        print("Pruning enabled")
    if regularization:
        print(f"Regularization enabled: {regularization}")
    
    log_cuda_memory("Before creating approximation dataset")
    
    # Create approximation dataset
    approx_train_dataset = create_dataset_from_teacher('ori_input', lenet_model, sample_points[0], batch_size, device, train_dataset)
    approx_test_dataset = create_dataset_from_teacher('ori_input', lenet_model, sample_points[1], batch_size, device, test_dataset)
    
    log_cuda_memory("After creating approximation dataset")
    
    results = []
    models_list = []
    
    for i, structure in enumerate(kan_structures):
        print(f"\nTesting KAN structure {i+1}/{len(kan_structures)}: {structure}")
        
        try:
            # Train KAN for approximation
            model, training_time, pruned_structure = train_kan_approximation(approx_train_dataset, approx_test_dataset, structure, device, batch_size, prune, regularization)
            
            # Evaluate approximation quality
            mse_error, max_error, mean_error = evaluate_kan_approximation(model, approx_test_dataset, device, batch_size)
            
            # Evaluate classification accuracy of approximated model
            accuracy = evaluate_kan_classification(model, test_dataset, device, batch_size)
            
            # Record results
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
            
            if prune and pruned_structure is not None:
                print(f"  Pruned Structure: {pruned_structure}")
                results.append({
                    "Method": "KAN Approximation (Pruned)",
                    "Structure": str(pruned_structure),
                    "Sample Points": sample_points,
                    "Training Time (s)": training_time,
                    "Classification Accuracy": accuracy,
                    "MSE": mse_error,
                    "Mean Absolute Error": mean_error,
                    "Maximum Absolute Error": max_error,
                    "Batch Size": batch_size,
                    "Pruned": True,
                    "Regularization": str(regularization) if regularization else "None",
                    "Original Structure": str(structure),
                    "Pruned Structure": str(pruned_structure)
                })
            
            meta = {
                "method": "KAN Approximation",
                "original_structure": str(structure),
                "pruned_structure": str(pruned_structure) if prune else "N/A",
                "prune": prune,
                "regularization": str(regularization) if regularization else "None",
                "accuracy": accuracy,
                "training_time": training_time,
                "batch_size": batch_size,
                "sample_points": sample_points
            }
            model_cpu = model.to('cpu')
            models_list.append({"meta": meta, "model": model_cpu})
            del model
            torch.cuda.empty_cache()
            
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
    
    return results, models_list


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
    device = torch.device('cuda:3')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Configuration parameters
    train_samples = 50000
    test_samples = 10000
    lenet_epochs = 100
    sample_points_list = [(50000, 10000)]
    batch_size = 2048
    
    # KAN structures
    kan_structures = [
        [784, 500, 100, 10],
        [784, 400, 200, 100, 50, 30, 10],
        [784, 300, 70, 10],
        [784, 200, 130, 60, 30, 10],
        [784, 100, 70, 10],
        [784, 100, 70, 50, 30, 10],
        [784, 50, 20, 10],
        [784, 30, 30, 20, 10],    
        [784, 20, 20, 20, 10, 10, 10],
    ]
    
    # Regularization configs
    regularization_configs = [
        None
    ]
    
    # Pruning configs
    pruning_configs = [False]
    
    log_cuda_memory("Before loading MNIST dataset")
    
    # Load MNIST dataset
    print("Loading MNIST dataset...")
    train_dataset, test_dataset = load_mnist_data(train_samples, test_samples, device)
    
    log_cuda_memory("After loading MNIST dataset")
    
    # Train LeNet-5
    print("\nTraining LeNet-5 for comparison...")
    lenet_model = train_lenet5(train_dataset, lenet_epochs, batch_size, device=device)
    lenet_accuracy = evaluate_model(lenet_model, test_dataset, batch_size, device=device)
    print(f"LeNet-5 Test Accuracy: {lenet_accuracy:.4f}")
    
    log_cuda_memory("After training LeNet-5")
    
    # Run benchmarks
    all_results = []
    direct_models = []
    approx_models = []
    
    for prune in pruning_configs:
        for regularization in regularization_configs:
            print(f"\n{'='*60}")
            print(f"Testing with prune={prune}, regularization={regularization}")
            print(f"{'='*60}")
    
            log_cuda_memory("Before running direct KAN training benchmark")
            
            # Direct KAN
            direct_results, direct_models_batch = run_direct_kan_benchmark(
                train_dataset, test_dataset, kan_structures, device, batch_size, prune, regularization
            )
            all_results.extend(direct_results)
            direct_models.extend(direct_models_batch)
    
            log_cuda_memory("After running direct KAN training benchmark")
            
            # Approximation with different sample points
            for sample_points in sample_points_list:
                print(f"\nRunning approximation benchmark with {sample_points} sample points...")
                approx_results, approx_models_batch = run_approximation_kan_benchmark(
                    train_dataset, test_dataset, lenet_model, kan_structures, sample_points, device, batch_size, prune, regularization
                )
                all_results.extend(approx_results)
                approx_models.extend(approx_models_batch)
    
    # Save results
    results_df = pd.DataFrame(all_results)
    
    results_dir = f"MNIST_results_{device}"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    results_file = os.path.join(results_dir, "benchmark_results.xlsx")
    results_df.to_excel(results_file, index=False)
    print(f"\nResults saved to: {results_file}")
    
    def get_model_accuracy(model_dict):
        return model_dict["meta"]["accuracy"]
    
    # Save best models
    if direct_models:
        best_direct = max(direct_models, key=get_model_accuracy)
        best_meta = best_direct["meta"]
        model = best_direct["model"]
        structure_str = best_meta["original_structure"].replace(" ", "")
        if best_meta["prune"] and best_meta["pruned_structure"] != "N/A":
            structure_str = f"original_{structure_str}_pruned_{best_meta['pruned_structure'].replace(' ', '')}"
        filename = f"best_direct_KAN_structure_lenet_accuracy_{lenet_accuracy:.4f}_{structure_str}_prune_{best_meta['prune']}_reg_{best_meta['regularization']}_accuracy_{best_meta['accuracy']:.4f}.pt"
        path = os.path.join("best_models", "MNIST", filename)
        os.makedirs("./best_models/MNIST", exist_ok=True)
        torch.save(model.state_dict(), path)
        print(f"Saved best direct model to {path}")
    
    if approx_models:
        best_approx = max(approx_models, key=get_model_accuracy)
        best_meta = best_approx["meta"]
        model = best_approx["model"]
        structure_str = best_meta["original_structure"].replace(" ", "")
        if best_meta["prune"] and best_meta["pruned_structure"] != "N/A":
            structure_str = f"original_{structure_str}_pruned_{best_meta['pruned_structure'].replace(' ', '')}"
        filename = f"best_approx_KAN_structure_lenet_accuracy_{lenet_accuracy:.4f}_{structure_str}_sample_points_{best_meta['sample_points']}_prune_{best_meta['prune']}_reg_{best_meta['regularization']}_accuracy_{best_meta['accuracy']:.4f}.pt"
        path = os.path.join("best_models", "MNIST", filename)
        os.makedirs("./best_models/MNIST", exist_ok=True)
        torch.save(model.state_dict(), path)
        print(f"Saved best approx model to {path}")
    
    # Print summary
    print("\n" + "="*50)
    print("BENCHMARK SUMMARY")
    print("="*50)
    print(f"LeNet-5 Baseline Accuracy: {lenet_accuracy:.4f}")
    print(f"Total experiments: {len(all_results)}")
    print(f"Results saved to: {results_file}")
    
    # Best results prints...
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
    
    # Overall best
    print("\nOverall Best Results:")
    if not direct_df.empty:
        overall_best_direct = direct_df.loc[direct_df['Classification Accuracy'].idxmax()]
        print(f"  Best Direct KAN: {overall_best_direct['Classification Accuracy']:.4f} accuracy")
        print(f"    Configuration: Prune={overall_best_direct['Pruned']}, Reg={overall_best_direct['Regularization']}")
        print(f"    Structure: {overall_best_direct['Structure']}")
        if overall_best_direct['Pruned'] and not direct_pruned_df.empty:
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
            pruned_match = approx_pruned_df[
                (approx_pruned_df['Original Structure'] == overall_best_approx['Structure']) &
                (approx_pruned_df['Regularization'] == overall_best_approx['Regularization']) &
                (approx_pruned_df['Sample Points'] == overall_best_approx['Sample Points'])
            ]
            if not pruned_match.empty:
                pruned_result = pruned_match.iloc[0]
                print(f"    Pruned Structure: {pruned_result['Structure']}")
    
    # Structure comparison
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