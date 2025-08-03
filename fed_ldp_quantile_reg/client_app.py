"""Federated LDP Quantile Regression Client Implementation."""

import torch
from scipy.stats import norm
import numpy as np
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from fed_ldp_quantile_reg.quantile_task import (
    QuantileNet,
    load_data,
    get_weights,
    set_weights,
    train
)

class QuantileClient(NumPyClient):
    def __init__(self, net, tau, r, trainloader, local_updates_mode , Em_list):
        self.net = net
        self.tau = tau
        self.r = r
        
        # params related to train data
        self.trainloader = trainloader
        self.local_updates_mode = local_updates_mode    # Em mode
        self.Em_list = Em_list         # contain Em detail
        self.server_rounds_cnt = 0     # used as Em idx

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)


    def fit(self, parameters, config):
        set_weights(self.net, parameters)

        train_loss = train(
            net=self.net,
            tau=self.tau,
            r=self.r,
            trainloader=self.trainloader,
            Em_list=self.Em_list,
            server_rounds_cnt=self.server_rounds_cnt,
            device=self.device,
        )
        self.server_rounds_cnt += 1

        return (
            get_weights(self.net),
            len(self.trainloader.dataset),
            {"train_loss": train_loss},
        )

    # def evaluate(self):
    #     """Evaluate the model using MSE"""
    #     tau = self.tau
    #     q_tau = norm.ppf(tau)
    #     beta_true = np.array([1 + q_tau] + [1] * 6)
    #     net = self.net
    #     with torch.no_grad():
    #         w = net.linear.weight.cpu().numpy()
    #         b = net.linear.bias.cpu().numpy()
    #         beta_pred = np.concatenate([[b], w])
    #         mse_total = np.mean((beta_pred - beta_true) ** 2)  # MSE

    #         # print("\n--- 最终模型参数 ---")
    #         # print(f"真实参数: {beta_true}")
    #         # print(f"估计参数: {beta_pred}")
    #         # print(f"整体 MSE: {mse_total:.6f}")
    #     return mse_total

def client_fn(context: Context):
    # Get configuration parameters
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    tau = context.run_config["tau"]
    r = context.run_config["r"]
    local_updates_mode = context.run_config["local-updates-mode"]   # Em mode

    # Load data for this partition
    trainloader, Em_list = load_data(
        partition_id=partition_id,
        num_partitions=num_partitions,
        context=context
    )

    # Return client instance
    net = QuantileNet()
    return QuantileClient(
        net=net,
        tau=tau,
        r=r,
        trainloader=trainloader,
        local_updates_mode=local_updates_mode,
        Em_list=Em_list
    ).to_client()


# Create Flower client
app = ClientApp(client_fn)
