import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
__package__ = "pykan"
from pykan.kan import *
import numpy as np
import pandas as pd 
import time
import torch
import itertools
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 必须导入


def func1(x):
    """f(x_1, x_2) = \frac{1}{2\pi * x_2} e^{-\frac{x_1^2}{2x_2}}

    Args:
        x (tuple of 2 dimensions)
    """
    
    # x1, x2 = x
    x1, x2 = x[:, 0], x[:, 1]
    return 1 / (2 * torch.pi * x2) * torch.exp(-x1**2 / (2 * x2))

def func2(x):
    """f(x_1, x_2) = \sqrt{1 + x_1^2 + 2x_1 cos x_2}

    Args:
        x (tuple of 2 dimensions)
    """
    
    x1, x2 = x[:, 0], x[:, 1]
    return torch.sqrt(1 + x1**2 + 2 * x1 * torch.cos(x2))

def func3(x):
    """f(x_1, x_2) = e^{x_1} * cos x_2

    Args:
        x (tuple of 2 dimensions)
    """
    
    x1, x2 = x[:, 0], x[:, 1]
    return torch.exp(x1) * torch.cos(x2)

def func4(x):
    """f(x_1, x_2) = e^{sin x_1} * cos x_2
    Args:
        x (tuple of 2 dimensions)
    """
    
    x1, x2 = x[:, 0], x[:, 1]
    return torch.exp(torch.sin(x1)) * torch.cos(x2)

def func5(x):
    """f(x_1, x_2, x_3) =  \frac{1}{2\pi * x_3} e^{-\frac{(x_1 - x_2)^2}{2x_3}}

    Args:
        x (tuple of 3 dimensions)
    """
    
    # x1, x2, x3 = x
    x1, x2, x3 = x[:, 0], x[:, 1], x[:, 2]
    return 1 / (2 * torch.pi * x3) * torch.exp(-((x1 - x2) ** 2) / (2 * x3))

def func6(x):
    """f(x_1, x_2, x_3) = \sqrt{1 + x_1^2 + 2x_1 cos (x_2 - x_3) }

    Args:
        x (tuple of 3 dimensions)
    """
    
    # x1, x2, x3 = x
    x1, x2, x3 = x[:, 0], x[:, 1], x[:, 2]
    return torch.sqrt(1 + x1**2 + 2 * x1 * torch.cos(x2 - x3))

def func7(x):
    """x_1 * \frac{sin^2 \frac{x_2 - x_3}{2}}{(\frac{x_2 - x_3}{2})^2}

    Args:
        x (tuple of 3 dimensions)
    """
    # x1, x2, x3 = x
    x1, x2, x3 = x[:, 0], x[:, 1], x[:, 2]
    diff = (x2 - x3) / 2  # 计算分母和正弦函数的输入

    # 使用 torch.where 处理分母为零的情况
    denominator = torch.where(diff == 0, torch.tensor(1.0, device=x.device), diff ** 2)
    numerator = torch.sin(diff) ** 2

    return x1 * numerator / denominator

def func8(x):
    """f(x_1, x_2, x_3) = x_1(1 + x_2cos x_3)

    Args:
        x (tuple of 3 dimensions)
    """
    # x1, x2, x3 = x
    x1, x2, x3 = x[:, 0], x[:, 1], x[:, 2]
    return x1 * (1 + x2 * torch.cos(x3))

def func9(x):
    """f(x_1, x_2, x_3, x_4, x_5, x_6) = \frac{x_1}{1 + (x_2 - 1)^2 + (x_3 - x_4)^2 + (x_5 - x_6)^2}

    Args:
        x (tuple of 6 dimensions)
    """
    
    # x1, x2, x3, x4, x5, x6 = x
    x1, x2, x3, x4, x5, x6 = x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4], x[:, 5]
    return x1 / (1 + (x2 - 1) ** 2 + (x3 - x4) ** 2 + (x5 - x6) ** 2)

def create_dataset_chebyshev(f, 
                   n_var=2, 
                   f_mode='col',
                   ranges=[-1, 1],
                   train_num=1000, 
                   test_num=1000,
                   normalize_input=False,
                   normalize_label=False,
                   device='cpu',
                   seed=0):
    '''
    create dataset with Chebyshev nodes for training samples
    
    Args:
    -----
        f : function
            the symbolic formula used to create the synthetic dataset
        ranges : list or np.array; shape (2,) or (n_var, 2)
            the range of input variables. Default: [-1,1].
        train_num : int
            the number of training samples. Default: 1000.
        test_num : int
            the number of test samples. Default: 1000.
        normalize_input : bool
            If True, apply normalization to inputs. Default: False.
        normalize_label : bool
            If True, apply normalization to labels. Default: False.
        device : str
            device. Default: 'cpu'.
        seed : int
            random seed. Default: 0.
        
    Returns:
    --------
        dataset : dic
            Train/test inputs/labels are dataset['train_input'], dataset['train_label'],
                        dataset['test_input'], dataset['test_label']
    '''
    np.random.seed(seed)
    torch.manual_seed(seed)

    if len(np.array(ranges).shape) == 1:
        ranges = np.array(ranges * n_var).reshape(n_var, 2)
    else:
        ranges = np.array(ranges)
    
    N = max(int(round(train_num ** (1.0 / n_var))), 5)  # 每个维度的点数
    print("points each dimension = ", N)
    
    nodes_list = []
    for i in range(n_var):
        # 生成 [-1,1] 上的 Chebyshev 节点
        nodes = np.cos((2 * np.arange(1, N + 1) - 1) / (2 * N) * np.pi)
        # 映射到对应的范围
        nodes_mapped = (ranges[i, 1] - ranges[i, 0]) / 2 * nodes + (ranges[i, 1] + ranges[i, 0]) / 2
        nodes_list.append(nodes_mapped)

    # 生成笛卡尔积，每个组合对应高维空间中的一个点
    cartesian_points = list(itertools.product(*nodes_list))
    train_input = torch.tensor(cartesian_points, dtype=torch.float32, device=device)
    print("train_input size = ", train_input.shape)
    
    # Generate random test samples
    test_input = torch.zeros(test_num, n_var)
    for i in range(n_var):
        test_input[:, i] = torch.rand(test_num) * (ranges[i, 1] - ranges[i, 0]) + ranges[i, 0]
    
    # Compute labels
    if f_mode == 'col':
        train_label = f(train_input)
        test_label = f(test_input)
    elif f_mode == 'row':
        train_label = f(train_input.T)
        test_label = f(test_input.T)
    else:
        raise ValueError(f'f_mode {f_mode} not recognized')

    # If labels have only 1 dimension, add an extra dimension
    if len(train_label.shape) == 1:
        train_label = train_label.unsqueeze(dim=1)
        test_label = test_label.unsqueeze(dim=1)
    
    def normalize(data, mean, std):
        return (data - mean) / std
    
    # Normalize inputs if required
    if normalize_input:
        mean_input = torch.mean(train_input, dim=0, keepdim=True)
        std_input = torch.std(train_input, dim=0, keepdim=True)
        train_input = normalize(train_input, mean_input, std_input)
        test_input = normalize(test_input, mean_input, std_input)
    
    # Normalize labels if required
    if normalize_label:
        mean_label = torch.mean(train_label, dim=0, keepdim=True)
        std_label = torch.std(train_label, dim=0, keepdim=True)
        train_label = normalize(train_label, mean_label, std_label)
        test_label = normalize(test_label, mean_label, std_label)
    
    # Create dataset dictionary
    dataset = {
        'train_input': train_input.to(device),
        'test_input': test_input.to(device),
        'train_label': train_label.to(device),
        'test_label': test_label.to(device)
    }
    
    return dataset

def kan_build(func, n_var, train_num, test_num, ranges, neurons=[5], sample_method="chebyshev"):
    # Build a KAN model for the given function
    
    torch.set_default_dtype(torch.float64)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # dataset = create_dataset(func, n_var=n_var, train_num=train_num, test_num=test_num, ranges=ranges, device=device)
    if sample_method == "chebyshev":
        # Create dataset using Chebyshev nodes
        dataset = create_dataset_chebyshev(func, n_var=n_var, train_num=train_num, test_num=test_num, ranges=ranges, device=device)
    elif sample_method == "random":
        # Create dataset using random sampling
        dataset = create_dataset(func, n_var=n_var, train_num=train_num, test_num=test_num, ranges=ranges, device=device)
        
    width = [n_var] + neurons + [1]
    model = KAN(width=width, grid=5, k=8, seed=2, device=device)
    
    loss_result = model.fit(dataset, opt="LBFGS", steps=20)
    
    return model, dataset

def model_evaluate(model, data, zero_mask = 1e-3):
    # evaluate the model on the dataset

    # obtain the errors
    model_predict = model.forward(data["test_input"])
    
    mse_error = torch.sqrt(torch.mean((model_predict - data["test_label"])**2))
    abs_error = torch.sqrt((model_predict - data["test_label"])**2)
    relative_error = torch.where(
        torch.abs(data["test_label"]) >= zero_mask,  # 条件：y_true >= zero_mask
        abs_error / torch.abs(data["test_label"]),  # 如果条件满足，计算相对误差
        abs_error  # 如果条件不满足，直接使用绝对误差
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

def visualize_approximation_3d(model, f, ranges, dims, steps=100, fixed_values=None, funcname="default", figpath="./approx.png"):
    """
    使用 3D 曲面图可视化高维函数近似效果（二维切片）。
    
    Args:
        model: 近似模型，接受形状为 (n_samples, n_var) 的输入。
        f: 真实函数，接受同样格式输入返回真实输出。
        ranges: list of tuples，每一维的输入范围，如 [(s,e), (s,e), ..., (s,e)]。
        dims: list，指定需要变化的二维索引（如 [d1, d2]）。
              根据此二维切片构造网格，其余维度固定。
        steps: int，每个变量的采样点数，默认 100。
        fixed_values: list，如果为 None，则各维取中值。
        title: str，图标题。
    """
    n_var = len(ranges)
    
    # 若未指定固定值，则各维取中值
    if fixed_values is None:
        fixed_values = [(r[0] + r[1]) / 2 for r in ranges]
    
    if len(dims) != 2:
        print("目前3D可视化仅支持二维切片，请指定 dims 长度为2。")
        return
    
    d1, d2 = dims[0], dims[1]
    x_vals = np.linspace(ranges[d1][0], ranges[d1][1], steps)
    y_vals = np.linspace(ranges[d2][0], ranges[d2][1], steps)
    X, Y = np.meshgrid(x_vals, y_vals)
    
    # 构造输入：二维变化，其他各维固定
    inputs = []
    for i in range(steps):
        for j in range(steps):
            inp = fixed_values.copy()
            inp[d1] = X[i, j]
            inp[d2] = Y[i, j]
            inputs.append(inp)
    inputs = torch.tensor(inputs, dtype=torch.float32, device=model.device)
    print("inputs shape: ", inputs.shape)
    
    # 获得真实函数值和模型预测值，reshape 到二维数组
    true_output = f(inputs).cpu().detach().numpy().reshape(steps, steps)
    model_output = model.forward(inputs).cpu().detach().numpy().reshape(steps, steps)
    
    abs_errors = np.abs(true_output - model_output)
    
    fig = plt.figure(figsize=(18, 6))

    # 第一个子图：真实函数曲面（统一蓝色）
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    surf_true = ax1.plot_surface(X, Y, true_output, color='blue', edgecolor='none', alpha=0.8)
    ax1.set_xlabel(f"Dimension {d1}")
    ax1.set_ylabel(f"Dimension {d2}")
    ax1.set_zlabel("Output")
    ax1.set_title("True Function")

    # 第二个子图：模型预测曲面（统一红色）
    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    surf_model = ax2.plot_surface(X, Y, model_output, color='red', edgecolor='none', alpha=0.8)
    ax2.set_xlabel(f"Dimension {d1}")
    ax2.set_ylabel(f"Dimension {d2}")
    ax2.set_zlabel("Output")
    ax2.set_title("Model Approximation")

    # 第三个子图：绝对误差曲面（采用 colormap 'inferno'）
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    surf_err = ax3.plot_surface(X, Y, abs_errors, cmap='inferno', edgecolor='none', alpha=0.8)
    ax3.set_xlabel(f"Dimension {d1}")
    ax3.set_ylabel(f"Dimension {d2}")
    ax3.set_zlabel("Absolute Error")
    ax3.set_title("Absolute Error")
    fig.colorbar(surf_err, ax=ax3, shrink=0.5, aspect=10, label="Error")

    fig.suptitle(funcname, fontsize=16)
    plt.tight_layout()
    plt.show()
    plt.savefig(figpath, bbox_inches='tight', dpi=300)
            
    return 

def dataset_visualize(dataset, f, ranges, funcname, figpath):
    """
    可视化二维输入数据集：
      - 第一个子图：散点图展示训练集输入点（平面图）。
      - 第二个子图：用训练集标签构成的3D曲面（蓝色），展示dataset记录的函数值。
      - 第三个子图：用真实函数f计算得到的值构成的3D曲面（红色）。
    
    Args:
        dataset: dict，包含 'train_input' 和 'train_label' （输入维度应为2）。
        f: 真实函数，使用 torch 实现，接收 tensor 输入。
        funcname: str，函数名称，用于图标题显示。
        figpath: str，保存图形的路径。
    """
    
    # 提取数据，并确保数据在 CPU 上
    train_input = dataset['train_input']
    train_label = dataset['train_label']
    if train_input.device.type != "cpu":
        train_input = train_input.cpu()
    if train_label.device.type != "cpu":
        train_label = train_label.cpu()

    # 转换为 numpy 数组
    inputs_np = train_input.detach().numpy()   # shape: (n_samples, 2)
    labels_np = train_label.detach().numpy().squeeze()  # shape: (n_samples,)
    
    # 计算真实函数 f 对应的值
    x_min, x_max = ranges[0]
    y_min, y_max = ranges[0]
    steps_grid = 100  # 可以根据需要调整步数
    x_vals = np.linspace(x_min, x_max, steps_grid)
    y_vals = np.linspace(y_min, y_max, steps_grid)
    X_grid, Y_grid = np.meshgrid(x_vals, y_vals)

    # 构造网格对应的输入（二维，每一行 [x,y]），并转换成 Torch 张量
    grid_inputs = np.stack([X_grid.ravel(), Y_grid.ravel()], axis=1)
    grid_tensor = torch.tensor(grid_inputs, dtype=torch.float32)  # 如果 f 在 GPU 上，请加 device=model.device

    # 计算真实函数 f 在网格上对应的输出，并 reshape 成 (steps_grid, steps_grid)
    true_vals_grid = f(grid_tensor).cpu().detach().numpy().reshape(steps_grid, steps_grid)


    # 创建 Figure 和三个子图
    fig = plt.figure(figsize=(18, 6))
    
    # 子图1：二维散点图，展示输入点
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.scatter(inputs_np[:, 0], inputs_np[:, 1], c='black', marker='o')
    ax1.set_title("Input Points")
    ax1.set_xlabel("Dimension 0")
    ax1.set_ylabel("Dimension 1")
    ax1.grid(True)
    
    # 子图2：用 dataset 的标签构成的 3D 曲面（蓝色）
    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    # 使用 plot_trisurf 绘制非结构化数据形成的曲面
    ax2.scatter(inputs_np[:, 0], inputs_np[:, 1], labels_np, color='blue', edgecolor='none', alpha=0.8)
    ax2.set_title("Training Set Surface")
    ax2.set_xlabel("Dimension 0")
    ax2.set_ylabel("Dimension 1")
    ax2.set_zlabel("Dataset Value")
    
    # 子图3：用真实函数 f 得到的值构成的 3D 曲面（红色）
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    ax3.plot_surface(X_grid, Y_grid, true_vals_grid, color='red', edgecolor='none', alpha=0.8)
    ax3.set_title("True Function Surface")
    ax3.set_xlabel("Dimension 0")
    ax3.set_ylabel("Dimension 1")
    ax3.set_zlabel("True f(x)")
    
    fig.suptitle(funcname, fontsize=16)
    plt.tight_layout()
    plt.savefig(figpath, bbox_inches='tight', dpi=300)
    plt.show()
    
    return

funcs2d = {
    "func1": {"f": lambda x: func1(x), "range": [(0.1, 1)] * 2},
    "func2": {"f": lambda x: func2(x), "range": [(0, 1)] * 2},
    "func3": {"f": lambda x: func3(x), "range": [(0, 1)] * 2},
    "func4": {"f": lambda x: func4(x), "range": [(0, 1)] * 2},
}

funcs3d = {
    "func5": {"f": lambda x: func5(x), "range": [(0.1, 1)] * 3},
    "func6": {"f": lambda x: func6(x), "range": [(0, 1)] * 3},
    "func7": {"f": lambda x: func7(x), "range": [(0, 1)] * 3},
    "func8": {"f": lambda x: func8(x), "range": [(0, 1)] * 3},
}

funcs6d = {
    "func9": {"f": lambda x: func9(x), "range": [(0, 1)] * 6},
}

test_cases = {
    "2d": (2, funcs2d),
    "3d": (3, funcs3d),
    "6d": (6, funcs6d),
}

if __name__ == "__main__":
    
    sample_method_list = ["chebyshev", "random"] # "chebyshev" or "random" training data sample method
    neurons_list = [3, 5, 7, 9] # one paramemter, middle layer neurons. Now, all the KAN is [n_var, middle_neurons, 1] structure. For each neuron, the parameter is (8 orders, 5 grids), which can be seen in function kan_build.
    
    for sample_method in sample_method_list:
        for middle_neurons in neurons_list:
            
            result_folder = f"./benchmark_results_{sample_method}/" # stores the visualization of 2d functions.
            figfolder = f"./benchmark_figures_{sample_method}/" # stores the test set errors (for accuracy), fitting time (sec), and the structures.
            
            # Ensure the result folder exists
            import os
            if not os.path.exists(result_folder):
                os.makedirs(result_folder)
            
            if not os.path.exists(figfolder):
                os.makedirs(figfolder)
                
            # middle_neurons = middle_neurons
            
            neurons = [middle_neurons]
            
            results = []
            
            for key, func_dict in test_cases.items():
                n_var, funcs = func_dict
                print(f"Running test cases for {key} functions:")
                
                for func_name, func_dict in funcs.items():
                    # print(f"Testing {func_name}...")
                    print(f"Running benchmark with sample method: {sample_method}, middle neurons: {middle_neurons}, on function {func_name} with {n_var} variables.")
                    ranges = func_dict["range"]
                    func = func_dict["f"]
                    
                    start = time.time()
                    model, data = kan_build(func, n_var=n_var, train_num=10000, test_num=1000, ranges=ranges, neurons=neurons, sample_method=sample_method)
                    end = time.time()

                    error_dict = model_evaluate(model, data, zero_mask=1e-6)
                    
                    # Visualize the approximation for 3D functions and the data.
                    if n_var <= 2:
                        visualize_approximation_3d(model, func, ranges, dims=[0, 1], steps=100, fixed_values=None, funcname=func_name, figpath=f"{figfolder}{func_name}_{middle_neurons}neurons_approximation.png")
                        
                        dataset_visualize(data, func, ranges, funcname=func_name, figpath=f"{figfolder}{func_name}_dataset_visualization.png")

                    results.append({
                        "n_var": n_var,
                        "Function": func_name,
                        "MSE": error_dict["MSE"],
                        "Mean Relative Error": error_dict["Mean Relative Error"],
                        "Maximum Relative Error": error_dict["Maximum Relative Error"],
                        "Time Taken (s)": end - start,
                        "layers": f"({n_var}, {neurons}, {1})"
                    })
                    
            # # Convert results to DataFrame for better visualization
            # results_df = pd.DataFrame(results)
            # results_df.to_excel(f"{result_folder}benchmark_results_{middle_neurons}neurons.xlsx", index=False)
            
            file_path = f"{result_folder}benchmark_results_{middle_neurons}neurons.xlsx"

            new_df = pd.DataFrame(results)
            if os.path.exists(file_path):
                existing_df = pd.read_excel(file_path)
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                combined_df.to_excel(file_path, index=False)
            else:
                new_df.to_excel(file_path, index=False)