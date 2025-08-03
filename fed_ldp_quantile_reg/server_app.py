"""test: A Flower / PyTorch app."""

from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from fed_ldp_quantile_reg.quantile_task import QuantileNet, get_weights
import numpy as np
from scipy.stats import norm

def get_evaluate_fn(tau):
    """Evaluate model parameters on the server."""
    
    q_tau = norm.ppf(tau)
    beta_true = np.array([1 + q_tau] + [1] * 6)
    
    def evaluate(server_round, parameters, config):
        weights = parameters[0]
        bias = parameters[1]
        
        beta_pred = np.concatenate([bias, weights.flatten()])
        
        mse = np.mean((beta_pred - beta_true) ** 2)
        
        print(f"\n--- 第{server_round}轮模型参数 ---")
        print(f"真实参数: {beta_true}")
        print(f"估计参数: {beta_pred}")
        print(f"整体 MSE: {mse:.6f}")
        
        return float(mse), {"mse": float(mse)}
    
    return evaluate

def server_fn(context: Context):
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]
    tau = context.run_config["tau"] 

    # Initialize model parameters
    ndarrays = get_weights(QuantileNet())
    parameters = ndarrays_to_parameters(ndarrays)

    # Define strategy
    strategy = FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0,
        min_available_clients=2,
        initial_parameters=parameters,
        evaluate_fn=get_evaluate_fn(tau),
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
