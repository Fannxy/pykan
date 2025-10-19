import torch
import math

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

from scipy.integrate import dblquad, tplquad
from scipy.stats import multivariate_normal
from scipy.stats import multivariate_t
from scipy.special import gamma
import math

def func10(x, a1=0, a2=0, b1=3, b2=2):
    """
    F(x,y) = \int_{a1}^x \int_{a2}^y \frac{1}{(b1-a1)(b2-a2)} , dv,du
    Args:
        x (tuple of 2 dimensions)
    """
    x1, x2 = x[:, 0], x[:, 1]
    batch_size = x1.shape[0]
    results = []

    def f(u, v):
        return 1 / (b1 - a1) * (b2 - a2)
    
    for i in range(batch_size):
        result, error = dblquad(f, a1, x1[i], a2, x2[i])
        results.append(result)

    results = torch.tensor(results)
    return results

def func11(x, rho=0.5):
    """
    F(x,y;rho) = \int_{-\infty}^x \int_{-\infty}^y
                 \frac{1}{2\pi \sqrt{1-\rho^2}}
                 \exp(-\frac{u^2 - 2\rho uv + v^2}{2(1-\rho^2)}) dv du

    Args:
        x (torch.Tensor): 一个形状为 (N, 2) 的张量，包含 N 个二维坐标点 (x, y)。
        rho (float): 相关系数，范围在 (-1, 1) 之间。
        
    Returns:
        torch.Tensor: 一个形状为 (N,) 的张量，包含每个输入点的 CDF 估计值。
    """
    if not -1 < rho < 1:
        raise ValueError("相关系数 rho 必须在 -1 和 1 之间。")

    x1, x2 = x[:, 0], x[:, 1]
    batch_size = x1.shape[0]
    results = []

    def f(u, v):
        return 1 / (2 * math.pi * math.sqrt(1 - rho**2)) * math.exp(-(u**2 - 2 * rho * u * v + v**2) / (2 * (1 - rho**2)))
    
    for i in range(batch_size):
        result, error = dblquad(f, -torch.inf, x1[i], -torch.inf, x2[i])
        results.append(result)

    results = torch.tensor(results)
    return results

def func12(x, lambda_1=1, lambda_2=1, lambda_3=1):
    """
    F(x,y,z) = \int_{0}^x \int_{0}^y \int_{0}^z
    \lambda_1 e^{-\lambda_1 u},\lambda_2 e^{-\lambda_2 v},\lambda_3 e^{-\lambda_3 w}
    , dw, dv, du
    Args:
        x (tuple of 3 dimensions)
    """
    x1, x2, x3 = x[:, 0], x[:, 1], x[:, 2]
    batch_size = x1.shape[0]
    results = []

    def f(u, v, w):
        return lambda_1 * math.exp(-lambda_1 * u) * lambda_2 * math.exp(-lambda_2 * v) * lambda_3 * math.exp(-lambda_3 * w)
    
    for i in range(batch_size):
        result, error = tplquad(f, 0, x1[i], 0, x2[i], 0, x3[i])
        results.append(result)

    results = torch.tensor(results)
    return results

def func13(x, mu=torch.tensor([0, 0, 0, 0]), cov=torch.eye(4), dtype=torch.float64):
    """
    F(\mathbf{x}) = \int_{-\infty}^{x_1} \cdots \int_{-\infty}^{x_4}
    \frac{1}{(2\pi)^{2} |\Sigma|^{1/2}}
    \exp!\left(-\tfrac{1}{2} (\mathbf{t}-\mu)^\top \Sigma^{-1} (\mathbf{t}-\mu)\right), d\mathbf{t}
    
    Args:
        x (torch.Tensor): 一个形状为 (N, 4) 的张量，包含 N 组积分上限点 (x1, x2, x3, x4)。
        mu (torch.Tensor): 均值向量，形状为 (4,)。
        cov (torch.Tensor): 协方差矩阵，形状为 (4, 4)。
        
    Returns:
        torch.Tensor: 一个形状为 (N,) 的张量，包含每个输入点的 CDF 估计值。
    """
    # 确保输入是 torch.Tensor, 并使用 float64 以提高精度
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float64)
    if not isinstance(mu, torch.Tensor):
        mu = torch.tensor(mu, dtype=torch.float64)
    if not isinstance(cov, torch.Tensor):
        cov = torch.tensor(cov, dtype=torch.float64)

    x = x.cpu().numpy()
    mu = mu.numpy()
    cov = cov.numpy()

    return torch.from_numpy(multivariate_normal.cdf(x, mu, cov)).to(torch.float32)

def func14(x, nu=5, mu=torch.tensor([0, 0, 0, 0, 0]), scale_matrix=torch.eye(5), dtype=torch.float64):
    """
    计算五维多元Student's t分布的累积分布函数 (CDF)。

    F(x) = ∫_{-∞}^{x_1} ... ∫_{-∞}^{x_5} pdf(t; nu, mu, Σ) dt
    
    其中 pdf(t; nu, mu, Σ) 是多元t分布的概率密度函数:
    pdf(t) = C * (1 + (1/ν) * (t-μ)ᵀ Σ⁻¹ (t-μ)) ^ (-(ν+d)/2)
    C = Γ((ν+d)/2) / [Γ(ν/2) * (νπ)^(d/2) * |Σ|^(1/2)]

    Args:
        x (torch.Tensor): 一个形状为 (N, 5) 的张量，包含 N 组积分上限点 (x1..x5)。
        nu (float): 自由度参数 (ν)。
        mu (torch.Tensor): 位置向量 (μ)，形状为 (5,)。
        scale_matrix (torch.Tensor): 尺度矩阵 (Σ)，形状为 (5, 5)。必须是正定的。
        
    Returns:
        torch.Tensor: 一个形状为 (N,) 的张量，包含每个输入点的 CDF 估计值。
    """
    # 确保输入是 torch.Tensor, 并使用 float64 以提高精度
    # 数值积分对精度非常敏感
    x = torch.as_tensor(x, dtype=torch.float64).cpu().numpy()
    mu = torch.as_tensor(mu, dtype=torch.float64).numpy()
    scale_matrix = torch.as_tensor(scale_matrix, dtype=torch.float64).numpy()
    nu = float(nu)
    
    return torch.from_numpy(multivariate_t.cdf(x, df=nu, loc=mu, shape=scale_matrix)).to(torch.float32)

def func15(x):
    """
    用 PyTorch 计算公式: x1 / (exp(x2/x3) + exp(-x2/x3))

    Args:
        x1 (torch.Tensor or float): 分子。
        x2 (torch.Tensor or float): 指数中的变量。
        x3 (torch.Tensor or float): 指数中的变量，不能为零。

    Returns:
        torch.Tensor: 计算结果。
    """
    x1, x2, x3 = x[:, 0], x[:, 1], x[:, 2]

    term_inside_exp = x2 / x3
    
    exp_term_pos = torch.exp(term_inside_exp)
    exp_term_neg = torch.exp(-term_inside_exp)
    denominator = exp_term_pos + exp_term_neg
    result = x1 / denominator
    
    return result

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math

def visualize_pytorch_function(
    func,
    domain,
    axes_to_plot,
    fixed_vars={},
    resolution=100,
    title="3D Function Visualization",
    save_path=None
):
    """
    可视化一个高维 PyTorch 函数的二维切片。

    Args:
        func (callable): 一个 PyTorch 函数，接收一个形状为 (N, D) 的 Tensor，
                         其中 N 是点的数量，D 是函数的总维度。
                         函数应返回一个形状为 (N,) 或 (N, 1) 的 Tensor。
        domain (dict): 一个字典，指定要绘制的维度的范围。
                       键(key)是维度索引(int)，值(value)是(min, max)元组。
                       例如: {0: (-5, 5), 1: (-5, 5)}
        axes_to_plot (tuple): 一个包含两个整数的元组，指定哪两个维度作为 x 和 y 轴。
                              例如: (0, 1) 表示使用第0维和第1维。
        fixed_vars (dict, optional): 一个字典，为非绘图维度指定固定值。
                                     键(key)是维度索引(int)，值(value)是其常数值。
                                     默认为 {}。
        resolution (int, optional): 每个轴上的采样点数。默认为 100。
        title (str, optional): 图像的标题。默认为 "3D Function Visualization"。
        save_path (str, optional): 保存图像的文件路径。如果为 None，则不保存。
                                   默认为 None。
    """
    if len(axes_to_plot) != 2:
        raise ValueError("axes_to_plot 必须包含两个维度索引。")

    # 1. 确定要绘制的两个维度
    x_ax_idx, y_ax_idx = axes_to_plot
    x_range = domain[x_ax_idx]
    y_range = domain[y_ax_idx]

    # 2. 创建二维网格点 (使用 NumPy)
    x_vals = np.linspace(x_range[0], x_range[1], resolution)
    y_vals = np.linspace(y_range[0], y_range[1], resolution)
    X, Y = np.meshgrid(x_vals, y_vals)

    # 3. 准备输入给 PyTorch 函数的张量
    # 确定函数的总维度
    all_dims = set(domain.keys()) | set(fixed_vars.keys())
    if not all_dims:
        total_dims = 0
    else:
        total_dims = max(all_dims) + 1

    # 创建一个 (resolution*resolution, total_dims) 的张量
    num_points = resolution * resolution
    points = torch.zeros(num_points, total_dims, dtype=torch.float32)

    # 填充变化的维度 (x 和 y 轴)
    # .flatten() 将二维网格展平为一维数组
    points[:, x_ax_idx] = torch.from_numpy(X.flatten())
    points[:, y_ax_idx] = torch.from_numpy(Y.flatten())

    # 填充固定的维度
    for dim_idx, value in fixed_vars.items():
        points[:, dim_idx] = value

    # 4. 使用 PyTorch 函数计算 Z 值
    # 在不计算梯度的模式下执行，以提高效率
    with torch.no_grad():
        Z_tensor = func(points)

    # 5. 将结果 Tensor 转换回用于绘图的 NumPy 数组
    # .reshape() 将一维结果转换回二维网格
    # .detach() 从计算图中分离
    # .cpu() 如果张量在GPU上，移动到CPU
    # .numpy() 转换为 NumPy 数组
    Z = Z_tensor.reshape(resolution, resolution).detach().cpu().numpy()

    # 6. 使用 Matplotlib 绘制 3D 表面图
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制表面
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', rstride=1, cstride=1)

    # 添加标签和标题
    ax.set_xlabel(f'Dimension {x_ax_idx}')
    ax.set_ylabel(f'Dimension {y_ax_idx}')
    ax.set_zlabel('Function Value')
    ax.set_title(title, fontsize=16)

    # 添加颜色条
    fig.colorbar(surf, shrink=0.5, aspect=10, label='Function Value')
    
    # 调整视角
    ax.view_init(elev=30., azim=125)
    plt.tight_layout()

    # 7. 保存图像并显示
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"图像已保存至: {save_path}")

    plt.show()
    plt.close(fig) # 释放内存

# --- 使用示例 ---

if __name__ == '__main__':
    visualize_pytorch_function(
        func=func14,
        domain={0: (0, 2), 1: (0, 2)}, 
        axes_to_plot=(0, 1), 
        fixed_vars={2:0, 3: 1, 4: 1},
        title="2D Function: function 14",
        save_path="Domain_figures/func14.png"
    )