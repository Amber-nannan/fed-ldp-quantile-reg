"""test: A Flower / PyTorch app."""

from flwr.common import Context, ndarrays_to_parameters,parameters_to_ndarrays
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from sympy import evaluate
from fed_ldp_quantile_reg.quantile_task import QuantileNet, get_weights
import numpy as np
from scipy.stats import norm
import torch

class FedPolyakRuppert(FedAvg):
    def __init__(self, tau, **kwargs):
        super().__init__(**kwargs)
        self.history_sum = None
        self.round_count = 0
        self.tau = tau

        q_tau = norm.ppf(tau)
        self.beta_true = np.array([1 + q_tau] + [1] * 6)

    def aggregate_fit(self, rnd, results, failures):
        """Perform FedAvg while maintain Polyak-Ruppert estimator."""

        # perform FedAvg aggregation to get the round's averaged parameters.
        aggregated_params, _ = super().aggregate_fit(rnd, results, failures)
        if aggregated_params is None:
            return None, {}

        # turn parameters into numpy
        aggregated_ndarrays = parameters_to_ndarrays(aggregated_params)

        # Polyak-Ruppert averaging
        if self.history_sum is None:
            self.history_sum = [np.array(p, copy=True) for p in aggregated_ndarrays]
        else:
            # print('self.history_sum',self.history_sum) 
            for i in range(len(self.history_sum)):
                self.history_sum[i] += aggregated_ndarrays[i]
            # print('self.history_sum',self.history_sum) 

        self.round_count += 1
        return aggregated_params, {}


    def evaluate(self, rnd, parameters):
        """Server-side evaluate: calculate MSE between PR estimator and true beta."""
        if self.history_sum is None or self.round_count == 0:
            return None
        
        # calculate PR estimator
        averaged_params = [p / self.round_count for p in self.history_sum]
        
        weights, bias = averaged_params
        beta_pred = np.concatenate([bias, weights.flatten()])
        mse = np.mean((beta_pred - self.beta_true) ** 2)

        return float(mse), {
            "mse": float(mse),
            "PR_estimator": [round(x, 6) for x in beta_pred.tolist()],
        }
        

def server_fn(context: Context):
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]
    tau = context.run_config["tau"] 
    seed = context.run_config['seed']

    # Initialize model parameters
    initial_net = QuantileNet()
    torch.manual_seed(seed) 
    with torch.no_grad(): 
        initial_net.linear.weight.copy_(torch.randn(1, 6)* 0.1)  # shape: [1, 6]
        initial_net.linear.bias.copy_(torch.tensor(0.0))
    ndarrays = get_weights(initial_net)
    # ndarrays = get_weights(QuantileNet())
    parameters = ndarrays_to_parameters(ndarrays)

    # Define strategy
    strategy = FedPolyakRuppert(
        tau=tau,
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0,
        min_available_clients=2,
        initial_parameters=parameters,
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
