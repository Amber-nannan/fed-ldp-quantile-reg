import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from scipy.stats import norm

def generate_data_example1(n, tau):
    """生成 Example 1 设定的数据。"""
    p = 6
    # 构建自相关协方差矩阵: σ_{j1,j2} = 0.5^{|j1-j2|}
    Sigma = np.array([[0.5 ** abs(j1 - j2) for j2 in range(p)] for j1 in range(p)])
    
    # 协变量 X ~ N(0, Sigma)
    X_covariates = np.random.multivariate_normal(mean=np.zeros(p), cov=Sigma, size=n)   # (n, 6)
    
    # 误差项 ε ~ N(0, 1)
    epsilon = np.random.randn(n)
    
    # 响应变量 Y = 1 + ΣX_j + ε
    y = 1 + np.sum(X_covariates, axis=1) + epsilon
    
    # 真实系数: β_τ = (1 + q_τ, 1, ..., 1)
    q_tau = norm.ppf(tau)
    beta_true = np.array([1 + q_tau] + [1] * 6)

    X = torch.from_numpy(X_covariates).float()
    y = torch.from_numpy(y).float()
    
    return X, y, beta_true


def lr_schedule(step,c0=0.01,a=0.51,b=0):
    """Learning rate schedule"""
    lr = c0 / (step**a + b)
    return lr


def LDP_QuantileReg(n, tau, r):
    torch.manual_seed(0)
    np.random.seed(0)

    X, y, beta_true = generate_data_example1(n, tau)

    # 创建 Dataset 和 DataLoader，batch_size=1 实现 SGD
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # LDP 参数设置
    tau_tilde = r * tau + (1 - r) / 2 

    # 模型 y = X @ w + b，初始化模型参数
    weights = torch.randn(6, requires_grad=False, dtype=torch.float32) * 0.1
    bias = torch.tensor(0.0, requires_grad=False, dtype=torch.float32)

    # 训练
    log_interval = 5000 
    step_count = 0


    for x_i, y_i in dataloader:
        x_i = x_i.squeeze(0)  # shape: (6,)
        y_i = y_i.squeeze()    # scalar
        
        y_pred_i = x_i @ weights + bias  # scalar
        z_true_i = (y_i <= y_pred_i).float()

        # LDP 扰动
        if torch.rand(1).item() < r:
            z_tilde_i = z_true_i
        else:
            z_tilde_i = torch.rand(1).item() < 0.5  # Bernoulli(0.5)
            z_tilde_i = torch.tensor(z_tilde_i, dtype=torch.float32)

        # 单样本梯度
        grad_weight = x_i * (z_tilde_i - tau_tilde)
        grad_bias = (z_tilde_i - tau_tilde)

        # 更新参数
        lr = lr_schedule(step_count//10 + 1)
        step_count += 1
        weights = weights - lr * grad_weight
        bias = bias - lr * grad_bias

        # # 打印loss
        # if step_count % log_interval == 0:
        #     with torch.no_grad():
        #         y_pred_full = X @ weights + bias
        #         residuals = y - y_pred_full
        #         pinball_loss = torch.mean(torch.where(residuals >= 0, tau * residuals, (tau - 1) * residuals))
        #         print(f"Step {step_count}, Pinball Loss: {pinball_loss.item():.4f}")


    # 评估结果
    with torch.no_grad():
        w = weights.numpy()
        b = bias.numpy()
        beta_pred = np.concatenate([[b], w])
        mse_total = np.mean((beta_pred - beta_true) ** 2)  # MSE over all 7 coefficients

        # print("\n--- 最终模型参数 ---")
        # print(f"真实参数: {beta_true}")
        # print(f"估计参数: {beta_pred}")
        # print(f"整体 MSE: {mse_total:.6f}")
    
    return mse_total


def run_experiment(taus, rs, Ns):
    results = []
    for tau in taus:
        for r in rs:
            for n in Ns:
                result = LDP_QuantileReg(n, tau, r)
                results.append(result)
                print(f"tau={tau}, r={r}, N={n} → MSE={result:.6f}")
    return results


if __name__ == "__main__":
    taus = [0.25, 0.5, 0.75]
    rs = [0.1, 0.5, 0.9]
    Ns = [1000, 5000, 10000, 50000, 100000]
    # Ns = [50000]
    results = run_experiment(taus, rs, Ns)