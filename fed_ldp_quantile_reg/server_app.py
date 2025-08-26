"""test: A Flower / PyTorch app."""

from flwr.common import Context, ndarrays_to_parameters,parameters_to_ndarrays
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from fed_ldp_quantile_reg.quantile_task import QuantileNet, get_weights
import numpy as np
from scipy.stats import norm


class FedPolyakRuppert(FedAvg):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.history_sum = None
        self.round_count = 0

    def aggregate_fit(self, rnd, results, failures):
        # perform FedAvg aggregation to get the round's averaged parameters.
        aggregated_params, _ = super().aggregate_fit(rnd, results, failures)
        if aggregated_params is None:
            return None, {}

        # turn parameters into numpy
        aggregated_ndarrays = parameters_to_ndarrays(aggregated_params)
        # print('#'*100)
        # print('aggregated_ndarrays',aggregated_ndarrays)

        # Polyak-Ruppert averaging
        if self.history_sum is None:
            self.history_sum = [np.array(p, copy=True) for p in aggregated_ndarrays]
        else:
            # print('self.history_sum',self.history_sum) 
            for i in range(len(self.history_sum)):
                self.history_sum[i] += aggregated_ndarrays[i]
            # print('self.history_sum',self.history_sum) 

        self.round_count += 1
        averaged_params = [p / self.round_count for p in self.history_sum]
        # print('averaged_params',averaged_params)
        return ndarrays_to_parameters(averaged_params), {}


def get_evaluate_fn(tau):
    """Evaluate model parameters on the server."""
    
    q_tau = norm.ppf(tau)
    beta_true = np.array([1 + q_tau] + [1] * 6)
    
    def evaluate(server_round, parameters, config):
        weights = parameters[0]
        bias = parameters[1]
        
        beta_pred = np.concatenate([bias, weights.flatten()])
        mse = np.mean((beta_pred - beta_true) ** 2)        
        
        return float(mse), {"round": server_round, "mse": float(mse), "beta_pred": [round(x, 6) for x in beta_pred.tolist()]}
    
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
    strategy = FedPolyakRuppert(
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
