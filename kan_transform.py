import os, sys
# 将 pykan 的父目录（即项目根目录）加入到 sys.path 中
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
__package__ = "pykan"

from pykan.benchmark import *

def obtain_b_splines(model, to_cpu=True):
    """Obtain the B-spline coefficients from the model."""

    b_splines = []
    dims_list = []
    for l in range(model.depth): # 2
        in_dim, out_dim = model.act_fun[l].coef.shape[0], model.act_fun[l].coef.shape[1]  # input dimension
        coeff_num = model.act_fun[l].coef.shape[-1]
        dims_list.append((in_dim, out_dim))
        # b_splines.append(b_splines_dict)
        b_spline_this_layer = []
        for i in range(in_dim):
            for j in range(out_dim):
                b_spline_this_layer.append({
                    "grid": model.act_fun[l].grid[0].cpu(),
                    "order": model.act_fun[l].k,
                    "coeff": model.act_fun[l].coef[i, j, :].detach().cpu(),
                    "mask": model.act_fun[l].mask[i, j].cpu(),
                    "scale_base": model.act_fun[l].scale_base[i, j].detach().cpu(),
                    "scale_sp": model.act_fun[l].scale_sp[i, j].detach().cpu(),
                })
        b_splines.append(b_spline_this_layer)
        
    return b_splines, dims_list

def evaluate_kan_neuron(x, base_act, b_spline_dict):
    x_act = base_act(x)
    y = evaluate_b_spline(b_spline_dict, x)
    
    # post process.
    y = b_spline_dict["scale_base"] * x_act + b_spline_dict["scale_sp"] * y
    
    return b_spline_dict["mask"] * y

def evaluate_b_spline(b_spline_dict, x):
    """
    用 NumPy 计算 B-spline 的输出值。
    
    参数:
      b_spline_dict: dict，包含键 "grid", "order", "coeff"，
                     - grid: 1D array, knot 序列
                     - order: int, B-spline 的阶数 k
                     - coeff: 1D array, 对应的系数，长度应为 len(grid) - (k+1)
      x: scalar 或者 1D array，评估点
    
    返回:
      f: 如果 x 为标量，则返回一个标量，
         如果 x 为数组，则返回一个同样长度的数组，值为 f(x)=∑_i coeff[i]*B_{i,k}(x)
    """

    # change to numpy array
    grid = np.array(b_spline_dict["grid"].cpu()).flatten()
    k = int(b_spline_dict["order"])
    coeff = np.array(b_spline_dict["coeff"].detach().cpu()).flatten()
    
    def bspline_basis(x_val, i, k, t):
        # 这个函数得到标量 x_val 对应的 B-spline 基函数 B_{i,k}(x_val) 的值
        # (递归实现，不做修改)
        if k == 0:
            if (t[i] <= x_val and x_val < t[i+1]) or (x_val == t[-1] and i == len(t)-2):
                return 1.0
            else:
                return 0.0
        else:
            denom1 = t[i+k] - t[i]
            term1 = 0.0 if denom1 == 0 else ((x_val - t[i]) / denom1) * bspline_basis(x_val, i, k-1, t)
                
            denom2 = t[i+k+1] - t[i+1]
            term2 = 0.0 if denom2 == 0 else ((t[i+k+1] - x_val) / denom2) * bspline_basis(x_val, i+1, k-1, t)
            return term1 + term2

    def bspline_basis_vectorized(x, i, k, t):
        """
        对于输入向量x（numpy 数组），返回每个 x 对应的 B-spline 基函数 B_{i,k}(x) 值
        """
        vec_func = np.vectorize(lambda xv: bspline_basis(xv, i, k, t))
        return vec_func(x)

    n_basis = len(grid) - (k + 1)
    results = np.zeros_like(x, dtype=float)
    for i in range(n_basis):
        b_vals = bspline_basis_vectorized(x, i, k, grid)  # shape 与 x 相同
        results += coeff[i] * b_vals
    return results


### test function.
def test_forward(x, base_act, extracted_b_splines, dims_list):
    
    x_layer = x
    
    for l, layer_bs in enumerate(extracted_b_splines):
        in_dim, out_dim = dims_list[l]
        # 假设每一层的 base function 存在于 model.act_fun[l].base_fun
        neuron_outputs = []
        
        for i in range(in_dim):
            for j in range(out_dim):
                b_dict = layer_bs[i * out_dim + j]
                x_val = x_layer[:, i]
                neuron_val = evaluate_kan_neuron(x_val, base_act, b_dict)
                # print("neuron_val: ", neuron_val)
                neuron_outputs.append(neuron_val)

        # 将 neuron_outputs 重组成 (batch, in_dim, out_dim)
        layer_tensor = torch.stack(neuron_outputs, dim=1).view(x_layer.shape[0], in_dim, out_dim)
        # print(layer_tensor.shape)
        # 原始 forward 对该层做的是 sum over in_dim：结果 shape (batch, out_dim)
        x_layer = torch.sum(layer_tensor, dim=1)
        # print(x_layer.shape)
        
    return x_layer


if __name__ == "__main__":
    
    model, data = kan_build(func1, n_var=2, train_num=1000, test_num=1000, ranges=[(0.1, 1), (0.1, 1)])
    
    test_input_gpu = torch.tensor([[0.5, 0.5], [0.2, 0.8]], dtype=torch.float64, device=model.device)
    test_input_cpu = torch.tensor([[0.5, 0.5], [0.2, 0.8]], dtype=torch.float64, device="cpu")


    b_splines_list, dims_list = obtain_b_splines(model)
    y_true = model.forward(test_input_gpu)
    y_test = test_forward(test_input_cpu, model.act_fun[0].base_fun, b_splines_list, dims_list)

    assert torch.allclose(y_test, y_true.cpu(), atol=1e-6), "The forward test failed, outputs do not match!"
    print("Forward test passed, outputs match!")