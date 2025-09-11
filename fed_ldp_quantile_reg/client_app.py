"""Federated LDP Quantile Regression Client Implementation."""

import torch
from scipy.stats import norm
import numpy as np
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context, ConfigRecord
from fed_ldp_quantile_reg.quantile_task import (
    QuantileNet,
    load_data,
    get_weights,
    set_weights,
    train
)

class QuantileClient(NumPyClient):
    def __init__(self, trainloader, Em_list, context: Context):
        self.net = QuantileNet()
        self.tau = context.run_config["tau"]
        self.r = context.run_config["r"]
        self.local_updates_mode = context.run_config["local-updates-mode"]    # Em mode
        
        # params related to train data
        self.trainloader = trainloader
        self.Em_list = Em_list         # contain Em detail

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)

        # record round info (stateful)
        self.client_state = (context.state)
        if 'round_info' not in context.state.config_records:
            context.state.config_records['round_info']  = ConfigRecord()


    def fit(self, parameters, config):
        set_weights(self.net, parameters)

        # update round_idx
        round_info = self.client_state.config_records['round_info'] 
        if 'round_idx' not in round_info:   
            round_info['round_idx'] = 0
        else:
            round_info['round_idx'] += 1

        train_loss = train(
            net=self.net,
            tau=self.tau,
            r=self.r,
            trainloader=self.trainloader,
            Em_list=self.Em_list,
            server_rounds_cnt=round_info['round_idx'],
            device=self.device,
        )

        return (
            get_weights(self.net),
            len(self.trainloader.dataset),
            {"train_loss": train_loss},
        )

def client_fn(context: Context):
    # Load data for this partition
    trainloader, Em_list = load_data(context=context)

    # Return client instance
    return QuantileClient(
        trainloader=trainloader,
        Em_list=Em_list,
        context=context
    ).to_client()


# Create Flower client
app = ClientApp(client_fn)
