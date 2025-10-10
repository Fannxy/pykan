from kan import *
import numpy as np
import pandas as pd 
import time
import torch
import datetime
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset, ConcatDataset
from torchvision import datasets, transforms
from torchvision import models
import matplotlib.pyplot as plt
import os
import random
import math

# 定义一个全局的日志文件名
LOG_FILE = "cuda_memory_log.txt"

# 如果日志文件已存在，先删除，以便每次运行都是新的记录
if os.path.exists(LOG_FILE):
    os.remove(LOG_FILE)

def log_cuda_memory(message, if_print=True):
    """
    记录当前CUDA显存使用情况到日志文件。
    
    Args:
        message (str): 描述当前代码位置或操作的自定义消息。
    """
    if not torch.cuda.is_available():
        print("CUDA is not available.")
        return
    
    if not if_print:
        return

    # 获取当前设备的显存使用情况
    # torch.cuda.memory_allocated(): 当前tensor分配的显存（字节）
    # torch.cuda.memory_reserved(): PyTorch缓存的显存（字节）
    allocated = torch.cuda.memory_allocated() / 1024**2  # 转换为 MB
    reserved = torch.cuda.memory_reserved() / 1024**2    # 转换为 MB
    
    # 获取时间戳
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
    
    # 格式化日志信息
    log_message = (
        f"[{timestamp}] {message}\n"
        f"  - Allocated: {allocated:.2f} MB\n"
        f"  - Reserved:  {reserved:.2f} MB\n"
        f"--------------------------------------------------\n"
    )
    
    # 追加写入到文件
    with open(LOG_FILE, "a") as f:
        f.write(log_message)

    # (可选) 如果你想更详细地分析，可以使用 memory_summary()
    # print(torch.cuda.memory_summary())


# -----------------------------------------------------------------------------
# ResNet-18 Implementation
# -----------------------------------------------------------------------------

class ResNet18(nn.Module):
    """ResNet-18 architecture for CIFAR-10 classification."""
    
    def __init__(self, num_classes=10):
        super(ResNet18, self).__init__()
        self.model = models.resnet18(num_classes=num_classes)
        
    def forward(self, x):
        return self.model(x)

# -----------------------------------------------------------------------------
# Data Loading and Preparation
# -----------------------------------------------------------------------------

def load_cifar10_data(train_samples=50000, test_samples=10000, device='cpu'):
    """
    Load CIFAR-10 dataset and prepare for KAN training.
    
    Args:
        device: Device to load data on
    
    Returns:
        dataset: Dictionary with train/test inputs and labels
    """
    print("Loading CIFAR-10 dataset...")
    if test_samples > 10000:
        test_samples = 10000
        print("test_samples is too large, set to 10000")
        
    cifar10_mean = (0.4914, 0.4822, 0.4465)
    cifar10_std = (0.2023, 0.1994, 0.2010)
    # Define normalization for CIFAR-10
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std)
    ])
    # 训练集的增广策略
    # 包括：随机裁剪、随机水平翻转、颜色抖动等
    transform_train_augmented = transforms.Compose([
        transforms.RandomCrop(32, padding=4),       # 在32x32图像周围填充4个像素，然后随机裁剪出32x32
        transforms.RandomHorizontalFlip(),            # 50%的概率水平翻转
        transforms.RandomRotation(30),                # 随机旋转-30到+30度
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), # 随机改变亮度、对比度、饱和度
        transforms.ToTensor(),                        # 将PIL Image或numpy.ndarray转换为tensor，并将像素值从[0, 255]缩放到[0, 1]
        transforms.Normalize(cifar10_mean, cifar10_std) # 用给定的均值和标准差进行归一化
    ])
    
    num_train_dataset = (train_samples + 49999)// 50000
    augmented_train_datasets = []
    for i in range(num_train_dataset):
        augmented_train_datasets.append(datasets.CIFAR10(root='../data', train=True, download=False, transform=transform_train_augmented))
    train_dataset_full = ConcatDataset(augmented_train_datasets)
    test_dataset_full = datasets.CIFAR10(root='../data', train=False, download=False, transform=transform_test)

    train_dataset = torch.utils.data.Subset(train_dataset_full, range(train_samples))
    test_dataset = torch.utils.data.Subset(test_dataset_full, range(test_samples))

    ori_train_dataset = datasets.CIFAR10(root='../data', train=True, download=False, transform=transform_test)
    ori_test_dataset = datasets.CIFAR10(root='../data', train=False, download=False, transform=transform_test)
    
    print(f"Dataset prepared: train_samples={train_samples}, test_samples={test_samples}")
    print(f"Device: {device}")
    
    return train_dataset, test_dataset, ori_train_dataset


def train_resnet18(dataset, epochs=30, batch_size=128, device='cpu'):
    """
    Train ResNet-18 model on CIFAR-10.
    
    Args:
        dataset: CIFAR-10 dataset
        epochs: Number of training epochs
        batch_size: Batch size for training
        device: Device to train on
    
    Returns:
        model: Trained ResNet-18 model
    """
    print(f"Training ResNet-18 for {epochs} epochs on {device} with batch_size={batch_size}...")
    
    model = ResNet18().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Load fresh dataset with augmentation
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
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
        
        if (epoch + 1) % 2 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}")
    
    return model


def evaluate_model(model, dataset, batch_size=256, device='cpu'):
    """
    Evaluate model accuracy on test set.
    
    Args:
        model: Trained model
        dataset: Test dataset
        device: Device to evaluate on
        batch_size: Batch size for evaluation
    
    Returns:
        accuracy: Classification accuracy
    """
    model.eval()
    
    # Load fresh test dataset
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
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


def apply_aug(batch_images):
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
    ])
    device = batch_images.device
    batch_images = batch_images.cpu()
    augmented = [transform(img) for img in batch_images]
    return torch.stack(augmented).to(device)


def train_kan_direct(train_dataset, test_dataset, structure, device='cpu', batch_size=256, prune=False, regularization=None):
    """
    Train KAN directly on CIFAR-10 data using batch processing.
    
    Args:
        train_dataset: CIFAR-10 train dataset
        test_dataset: CIFAR-10 test dataset
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
    model = KAN(width=structure, grid=5, k=3, seed=42, device=device, auto_save=False)

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
        
        loss_result = model.fit(
            batch_dataset, 
            opt="LBFGS", 
            loss_fn=nn.CrossEntropyLoss(),
            steps=20,
            batch=batch_size,
            lamb=lamb,
            lamb_l1=lamb_l1,
            lamb_entropy=lamb_entropy,
            lamb_coef=lamb_coef,
            lamb_coefdiff=lamb_coefdiff,
            reg_metric=reg_metric
        )

    # Apply pruning if requested
    if prune:
       pass
    
    training_time = time.time() - start_time
    
    return model, training_time, pruned_structure

class GeneratedDataset(Dataset):
    """
    一个由教师模型生成标签的数据集。
    """
    def __init__(self, inputs, labels):
        # 确认输入和标签的数量一致
        assert len(inputs) == len(labels), "输入和标签的数量必须相同"
        self.inputs = inputs
        self.labels = labels

    def __len__(self):
        """返回数据集中样本的总数"""
        return len(self.inputs)

    def __getitem__(self, idx):
        """根据索引idx获取一个样本（输入和对应的标签）"""
        input_sample = self.inputs[idx]
        label_sample = self.labels[idx]
        return input_sample, label_sample

class WeightedCEL_KLDLoss(nn.Module):
    """
    一个健壮的自定义损失函数，计算交叉熵损失和KL散度损失的加权和。
    
    能够自动处理两种y_true格式：
    1. 类别索引 (long tensor)，形状为 [batch_size]。
    2. 类别分布，如one-hot或软标签 (float tensor)，形状为 [batch_size, num_classes]。
    """
    def __init__(self, alpha=1.0, temperature=1.0):
        super(WeightedCEL_KLDLoss, self).__init__()
        if not 0 <= alpha <= 1:
            raise ValueError("权重 alpha 必须在 [0, 1] 范围内。")
        self.alpha = alpha
        self.temperature = temperature
        
        # 用于计算KL散度，输入是log-prob，目标是prob
        self.kl_div_loss = nn.KLDivLoss(reduction='batchmean')
        # 用于计算标准的交叉熵（当y_true是索引时）
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, y_pred, y_true):
        """
        计算加权损失。
        
        参数:
        - y_pred (torch.Tensor): 模型的原始输出 (logits)，形状 [batch_size, num_classes]。
        - y_true (torch.Tensor): 真实标签，可以是索引或分布。
        """
        num_classes = y_pred.shape[1]
        
        y_true_dist = F.softmax(y_true, dim=1)
        # 1.b 计算交叉熵损失 (手动计算，以支持软标签)
        # CE(p, q) = - sum(p_i * log(q_i))
        # F.log_softmax(y_pred) 得到 log(q_i)
        log_pred_softmax = F.log_softmax(y_pred / self.temperature, dim=1)
        ce_loss = -torch.sum(y_true_dist * log_pred_softmax, dim=1).mean()

        # 2. 计算KL散度损失
        kl_loss = self.kl_div_loss(log_pred_softmax, y_true_dist)
        
        # 3. 计算加权总损失
        total_loss = self.alpha * ce_loss + (1 - self.alpha) * kl_loss
        print(f"ce_loss: {ce_loss}, kl_loss: {kl_loss}, total_loss: {total_loss}")
        
        return total_loss

# --- 步骤 2: 编写生成数据和标签的函数 ---
def create_dataset_from_teacher(strategy, teacher_model, sample_points, batch_size, device, ori_dataset=None, chebyshev_degree=10):
    """
    使用教师模型生成数据集。

    Args:
        strategy (str): 生成数据集的策略，'random' 或 'chebyshev' 或 'ori_input'。
        teacher_model (nn.Module): 预训练的教师模型。
        sample_points (int): 要生成的总样本数。
        batch_size (int): 每次生成样本的批次大小。
        device (torch.device): 运行模型的设备 ('cuda' or 'cpu')。
    
    Returns:
        GeneratedDataset: 包含生成数据和标签的PyTorch Dataset对象。
    """
    print(f"开始生成 {sample_points} 个样本...")
    
    # 将模型移至指定设备并设置为评估模式
    teacher_model.eval()

    all_inputs = []
    all_labels = []

    # 使用 torch.no_grad()，因为我们只是在做推理，不需要计算梯度
    with torch.no_grad():
        ori_dataloader = None
        if strategy == 'ori_input':
            ori_dataloader = DataLoader(ori_dataset, batch_size=batch_size, shuffle=False)

        # 计算需要多少批次
        num_batches = (sample_points + batch_size - 1) // batch_size
        
        for _ in tqdm(range(num_batches), desc="生成数据"):
            # 1. 生成输入数据
            if strategy == 'random':
                inputs = torch.rand(batch_size, 3, 32, 32, device=device)
            elif strategy == 'chebyshev':
                def generate_chebyshev_tensor(batch_size: int, n: int) -> torch.Tensor:
                    """
                    随机生成一个[batch_size, 3, 32, 32]的张量，其值为[0,1]上的切比雪夫点。

                    Args:
                        batch_size (int): 批处理大小，张量的第一个维度。
                        n (int): 切比雪夫点的总数（阶数+1）。为了使公式有意义，n必须大于等于2。
                    Returns:
                        torch.Tensor: 一个形状为[batch_size, 3, 32, 32]的浮点数张量， 
                    其元素是根据随机整数索引生成的切比雪夫点。
                    """
                    # 检查n的有效性，因为公式中包含 n-1 作为分母
                    if n < 2:
                        raise ValueError("n 必须大于等于 2，才能定义不同的切比雪夫点。")
                    indices = torch.randint(0, n, (batch_size, 3, 32, 32))
                    indices_float = indices.float()
                    chebyshev_points = (torch.cos(indices_float * math.pi / (n - 1)) + 1) / 2.0
                    return chebyshev_points

                inputs = generate_chebyshev_tensor(batch_size, chebyshev_degree).to(device)
            elif strategy == 'ori_input':
                try:
                    inputs, _ = next(iter(ori_dataloader))
                    # if inputs.shape[0] != batch_size:
                    #     added_inputs = torch.rand(batch_size - inputs.shape[0], 3, 32, 32, device=device)
                    #     inputs = torch.cat([inputs.to(device), added_inputs], dim=0)
                    # else:
                    #     inputs = inputs.to(device)
                    inputs = inputs.to(device)
                except StopIteration:
                    # inputs = torch.rand(batch_size, 3, 32, 32, device=device)
                    break
            else:
                raise ValueError(f"Invalid strategy: {strategy}")
            
            # 2. 获取教师模型的输出作为标签
            labels = teacher_model(inputs)
            
            # 3. 将生成的输入和标签移回CPU并存储
            # 将数据保存在CPU上可以避免在不需要时占用GPU内存
            all_inputs.append(inputs.cpu())
            all_labels.append(labels.cpu())

    # 将列表中的所有张量拼接成一个大张量
    final_inputs = torch.cat(all_inputs, dim=0)
    final_labels = torch.cat(all_labels, dim=0)

    # 有可能生成的样本数略多于sample_points（因为最后一个batch）
    # 这里我们精确截取所需的数量
    final_inputs = final_inputs[:sample_points]
    final_labels = final_labels[:sample_points]
    
    print(f"生成完成！最终数据集大小: inputs.shape={final_inputs.shape}, labels.shape={final_labels.shape}")

    # 4. 用生成的数据创建Dataset实例
    return GeneratedDataset(final_inputs, final_labels)

def train_kan_approximation(train_dataset, test_dataset, structure, device='cpu', batch_size=256, prune=False, regularization=None):
    """
    Train KAN to approximate ResNet-18 using batch processing.
    
    Args:
        train_dataset: Approximation train dataset
        test_dataset: Approximation test dataset
        structure: KAN architecture
        device: Device to train on
        batch_size: Batch size for training to handle GPU memory
        prune: Whether to apply pruning after training and retrain
        regularization: Dictionary with regularization parameters
    
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

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    # Create KAN model
    model = KAN(width=structure, grid=5, k=3, seed=42, device=device, auto_save=False)

    
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
        
        # Train on this batch with regularization
        loss_result = model.fit(
            batch_dataset, 
            opt="LBFGS", 
            steps=20,
            loss_fn=WeightedCEL_KLDLoss(alpha=0.5),
            batch=batch_size,
            lamb=lamb,
            lamb_l1=lamb_l1,
            lamb_entropy=lamb_entropy,
            lamb_coef=lamb_coef,
            lamb_coefdiff=lamb_coefdiff,
            reg_metric=reg_metric
        )
        

    # Apply pruning if requested
    if prune:
        pass
    
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


def evaluate_kan_approximation(model, dataset, device='cpu', batch_size=256):
    """
    Evaluate KAN approximation of ResNet-18.
    
    Args:
        model: Trained KAN model
        dataset: Test dataset
        resnet_model: Original ResNet-18 model
        device: Device to evaluate on
    
    Returns:
        mse_error: Mean squared error
        max_error: Maximum absolute error
        mean_error: Mean absolute error
    """
    model.eval()
    
    with torch.no_grad():
        test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
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

# -----------------------------------------------------------------------------
# Main Benchmark Functions
# -----------------------------------------------------------------------------

def run_direct_kan_benchmark(train_dataset, test_dataset, kan_structures, device='cpu', batch_size=256, prune=False, regularization=None):
    """
    Run benchmark for direct KAN training on CIFAR-10.
    
    Args:
        dataset: CIFAR-10 dataset
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
    models_list = []
    
    for i, structure in enumerate(kan_structures):
        print(f"\nTesting KAN structure {i+1}/{len(kan_structures)}: {structure}")
        
        try:
            # Train KAN directly
            model, training_time, pruned_structure = train_kan_direct(train_dataset, test_dataset, structure, device, batch_size, prune, regularization)
            
            # Evaluate classification accuracy
            accuracy = evaluate_kan_classification(model, test_dataset, device)

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
            
            # If pruning was applied, add an additional row for the pruned structure
            if prune and pruned_structure is not None:
                print(f"  Pruned Structure: {pruned_structure}")
                # Add a separate row for the pruned structure
                results.append({
                    "Method": "Direct KAN (Pruned)",
                    "Structure": str(pruned_structure),
                    "Training Time (s)": training_time,  # Same training time since it's the same model
                    "Classification Accuracy": accuracy,  # Same accuracy since it's the same model
                    "MSE": "N/A",  # Same MSE since it's the same model
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


def run_approximation_kan_benchmark(train_dataset, test_dataset, resnet_model, kan_structures, sample_points=(10000, 1000), device='cpu', batch_size=256, prune=False, regularization=None):
    """
    Run benchmark for KAN approximation of ResNet-18.
    
    Args:
        dataset: CIFAR-10 dataset
        resnet_model: Trained ResNet-18 model
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
    print("KAN APPROXIMATION OF RESNET-18 BENCHMARK")
    print("="*50)
    if prune:
        print("Pruning enabled")
    if regularization:
        print(f"Regularization enabled: {regularization}")
    
    # Create approximation dataset
    approx_train_dataset = create_dataset_from_teacher('ori_input', resnet_model, sample_points[0], batch_size, device, train_dataset)
    approx_test_dataset = create_dataset_from_teacher('ori_input', resnet_model, sample_points[1], batch_size, device, test_dataset)
    
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
            accuracy = evaluate_kan_classification(model, test_dataset, device)
            
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
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cuda:3')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Configuration parameters
    train_samples = 500000  # Number of training samples to use
    test_samples = 10000    # Number of test samples to use
    resnet_epochs = 200      # Number of epochs for ResNet-18 training
    sample_points_list = [(500000, 10000)]  # Different numbers of sample points for approximation, the first is for training, the second is for testing
    batch_size = 2048        # Batch size for KAN training to handle GPU memory
    
    # KAN structures to test (input_dim = 3072, output_dim = 10)
    kan_structures = [
        [3072, 100, 50, 30, 10],
        [3072, 200, 70, 10],
        [3072, 150, 50, 10],
        [3072, 100, 30, 10],
        [3072, 50, 30, 10]
    ]
    
    # Regularization configurations to test
    regularization_configs = [
        None  # No regularization
    ]
    
    # Pruning configurations to test
    pruning_configs = [False]
    
    # Load CIFAR-10 dataset
    print("Loading CIFAR-10 dataset...")
    train_dataset, test_dataset, ori_train_dataset = load_cifar10_data(train_samples, test_samples, device)

    # Train ResNet-18 for comparison
    print("\nTraining ResNet-18 for comparison...")
    resnet_model = train_resnet18(ori_train_dataset, resnet_epochs, batch_size, device)
    resnet_accuracy = evaluate_model(resnet_model, test_dataset, batch_size, device)
    print(f"ResNet-18 Test Accuracy: {resnet_accuracy:.4f}")
    
    # Run benchmarks
    all_results = []
    direct_models = []
    approx_models = []
    
    # Test different combinations of pruning and regularization
    for prune in pruning_configs:
        for regularization in regularization_configs:
            print(f"\n{'='*60}")
            print(f"Testing with prune={prune}, regularization={regularization}")
            print(f"{'='*60}")
            
            # # 1. Direct KAN training benchmark
            # direct_results, direct_models_batch = run_direct_kan_benchmark(
            #     train_dataset, test_dataset, kan_structures, device, batch_size, prune, regularization
            # )
            # all_results.extend(direct_results)
            # direct_models.extend(direct_models_batch)
            
            # 2. KAN approximation benchmark with different sample points
            for sample_points in sample_points_list:
                print(f"\nRunning approximation benchmark with {sample_points} sample points...")
                approx_results, approx_models_batch = run_approximation_kan_benchmark(
                    train_dataset, test_dataset, resnet_model, kan_structures, sample_points, device, batch_size, prune, regularization
                )
                all_results.extend(approx_results)
                approx_models.extend(approx_models_batch)
    
    # Save results
    results_df = pd.DataFrame(all_results)
    
    # Create results directory
    results_dir = f"CIFAR10_results_{device}"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # Save to Excel file
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
        filename = f"best_direct_KAN_structure_{structure_str}_prune_{best_meta['prune']}_reg_{best_meta['regularization']}_accuracy_{best_meta['accuracy']:.4f}.pt"
        path = os.path.join("best_models", "CIFAR10", filename)
        os.makedirs("./best_models/CIFAR10", exist_ok=True)
        torch.save(model.state_dict(), path)
        print(f"Saved best direct model to {path}")

    if approx_models:
        best_approx = max(approx_models, key=get_model_accuracy)
        best_meta = best_approx["meta"]
        model = best_approx["model"]
        structure_str = best_meta["original_structure"].replace(" ", "")
        if best_meta["prune"] and best_meta["pruned_structure"] != "N/A":
            structure_str = f"original_{structure_str}_pruned_{best_meta['pruned_structure'].replace(' ', '')}"
        filename = f"best_approx_KAN_structure_{structure_str}_sample_points_{best_meta['sample_points']}_prune_{best_meta['prune']}_reg_{best_meta['regularization']}_accuracy_{best_meta['accuracy']:.4f}.pt"
        path = os.path.join("best_models", "CIFAR10", filename)
        os.makedirs("./best_models/CIFAR10", exist_ok=True)
        torch.save(model.state_dict(), path)
        print(f"Saved best approx model to {path}")
    
    # Print summary
    print("\n" + "="*50)
    print("BENCHMARK SUMMARY")
    print("="*50)
    print(f"ResNet-18 Baseline Accuracy: {resnet_accuracy:.4f}")
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