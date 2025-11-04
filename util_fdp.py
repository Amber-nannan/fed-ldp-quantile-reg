import sys
import time
sys.path.append('..')
from util import *
from FedDPQuantile import FedDPQuantile
from DPQuantile import DPQuantile
import pickle
import os
from scipy.stats import norm
from scipy.optimize import root_scalar
import math

def generate_lists(start, end, K):
    list_start = [start] * K
    list_avg = np.linspace(start, end, K).tolist()
    list_end = [end] * K
    return list_start, list_avg, list_end

def objective(x, mu_list, taus):
    """Global objective function: Calculate (1/K) * sum(Φ(x - μ_k)) - τ (σ=1)"""
    return np.mean(norm.cdf(x - np.asarray(mu_list))) - np.mean(taus)

def solve_quantile(mu_list, taus, xtol=1e-8):
    """Solve equation: (1/K) * sum(Φ(x - μ_k)) = τ (σ=1, uniform weights)"""
    # Determine search interval (covering 99.99994% probability range)
    mu_min, mu_max = np.min(mu_list), np.max(mu_list)
    bracket = (mu_min - 5, mu_max + 5)
    # Call the external target function with bound parameters using lambda
    result = root_scalar(lambda x: objective(x, mu_list, taus),
                         method='bisect',bracket=bracket,xtol=xtol)
    return result.root

def _gen_power_seq(alpha, max_len):
    """Generate [⌈1^α⌉, ⌈2^α⌉, …, ⌈max_len^α⌉]"""
    return [int(math.ceil((i+1)**alpha)) for i in range(max_len)]

def get_Em_list(T, warm_up=0.05, typ='log', E_cons=1, T_mode='rounds'):
    """
    Generate the list of local iteration counts
    
    Returns:
        Total sample size, iteration counts list
    """
    if T_mode == 'rounds':    # Based on communication rounds
        pre = int(T * warm_up)
        minor = T - pre
        if typ == 'log':
            Em_minor = [int(np.ceil(np.log2(m+1))) for m in range(1, minor+1)]
        elif typ == 'cons':
            Em_minor = [E_cons] * minor
        elif isinstance(typ, (int, float)) and typ > 0:
            # Power growth
            Em_minor = _gen_power_seq(typ, minor)
        else:
            raise ValueError("typ must be 'log', 'cons', or a float in (0,1].")
        return pre + sum(Em_minor), [1] * pre + Em_minor
    
    elif T_mode == 'samples':     # Based on total sample size
        total_samples   = T
        warm_up_samples = int(total_samples * warm_up)
        remaining       = total_samples - warm_up_samples

        if typ == 'cons':
            rounds, leftover = divmod(remaining, E_cons)
            Em_list = [1]*warm_up_samples + [E_cons]*rounds
            if leftover > 0:
                Em_list.append(leftover)
            return total_samples, Em_list
            
        elif typ == 'log':
            Em, cur, n = [], 0, 2         # Start from log₂(2)
            while cur + math.ceil(math.log2(n)) < remaining:
                step = math.ceil(math.log2(n))
                Em.append(step); cur += step; n += 1
            if cur < remaining:
                Em.append(remaining - cur)
            return total_samples, [1]*warm_up_samples + Em

        elif isinstance(typ, (int, float)) and typ > 0:
            Em, cur, n = [], 0, 1
            while cur + math.ceil(n**typ) < remaining:
                step = math.ceil(n**typ)
                Em.append(step); cur += step; n += 1
            if cur < remaining:
                Em.append(remaining - cur)
            return total_samples, [1]*warm_up_samples + Em
        else:
            raise ValueError("typ must be 'log', 'cons', or a float in (0,1].")
    else:
        raise ValueError("T_mode must be 'rounds' or 'samples'.")

def save_pickle(var, file_path):
    """Save variables to pickle file"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'wb') as f:
        pickle.dump(var, f)

def load_pickle(file_path):
    """Load variables from pickle file"""
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def train(seed, dist_type, taus, client_rs, n_clients, T, E_typ='log', E_cons=1,
          gene_process='homo', mode='federated', use_true_q_init=False, a=0.51, b=100,c=2,
          return_history=False,T_mode='rounds'):
    """
    Single federated experiment
    
    Params:
        mode: 'federated' (federated training) or 'global' (global training)
        T_mode: 'rounds' (based on communication rounds) or 'samples' (based on total sample size)
    """
    taus = [taus] * n_clients if isinstance(taus, (int, float)) else taus
    np.random.seed(42)
    # mus = np.random.randn(n_clients) if gene_process == 'hete' else np.zeros(n_clients)
    if gene_process == 'hete':
        mus = np.random.randn(n_clients)
    elif gene_process == 'homo' or gene_process=='hete_d':
        mus = np.zeros(n_clients)
    elif isinstance(gene_process,float):
        mus = np.random.normal(loc=0,scale=gene_process,size=n_clients)
        
    clients_data = []
    ET, Em_list = get_Em_list(T, typ=E_typ, E_cons=E_cons, T_mode=T_mode)

    if gene_process != 'hete_d':
        for k in range(n_clients):
            data, _ = generate_data(dist_type, taus[k], ET, mu=mus[k])    # ET samples
            clients_data.append(data)
        global_true_q = solve_quantile(mus, taus)

    elif gene_process == 'hete_d':
        # Three distribution types: normal, uniform, cauchy
        distribution_pool = ['normal', 'uniform', 'cauchy']
        
        # Divide n_clients among three distributions as evenly as possible
        c1 = n_clients // 3
        c3 = n_clients - 2*c1
        # Get dist_list, e.g., for 10 machines => ['normal','normal','normal', ... 'uniform' x3, 'cauchy' x4]
        dist_list = []
        dist_list += [distribution_pool[0]] * c1
        dist_list += [distribution_pool[1]] * c1
        dist_list += [distribution_pool[2]] * c3
        
        # Generate corresponding distribution data for each client
        for k in range(n_clients):
            data, _ = generate_data(dist_list[k], taus[k], ET, mu=mus[k])
            clients_data.append(data)

        global_true_q = 0.0  # Can only accept median
    
    # Choose data processing mode
    if mode == 'global':
        # Global training mode: merge data, n_clients=1
        Q_avgs = []; Vars = []; History = {}
        for i,data_i in enumerate(clients_data):
            model = DPQuantile(tau=taus[i], r=client_rs[i], true_q=global_true_q,a=a, b=b,c=c,seed=seed)
            model.fit(data_i)
            Q_avgs.append(model.Q_avg)
            Vars.append(model.get_variance())
            History[i]=model.get_stats_history()
        if return_history:
            return global_true_q, History
        else:
            return global_true_q, np.mean(Q_avgs), np.mean(Vars), _

    elif mode == 'federated':
        # Federated training mode: keep client data separate
        model = FedDPQuantile(n_clients=n_clients, client_rs=client_rs,
                              taus=taus,
                              true_q=global_true_q,use_true_q_init=use_true_q_init,a=a, b=b,c=c,seed=seed)
        model.fit(clients_data, Em_list)

    if return_history:
        return global_true_q, model.get_stats_history()
    else:
        return global_true_q, model.Q_avg, model.get_variance(), model.errors


@ray.remote
def train_remote(seed, dist_type, taus, client_rs, n_clients, T, E_typ,
                 E_cons,gene_process,mode,use_true_q_init=False,a=0.51,
                 b=100,c=2,T_mode='rounds'):
    return train(seed, dist_type, taus, client_rs, n_clients, T, E_typ,
                 E_cons,gene_process,mode,use_true_q_init=use_true_q_init,a=a, b=b,c=c,
                T_mode=T_mode)


def run_federated_simulation(dist_type, taus, client_rs, n_clients, 
                            T,E_typ, E_cons,gene_process, mode, n_sim,use_true_q_init=False, base_seed=2025,
                            a=0.51, b=100,c=2,T_mode='rounds'):

    futures = [train_remote.remote(base_seed + i,
            dist_type, taus, client_rs, n_clients, T,
                                   E_typ, E_cons, gene_process, mode,use_true_q_init=use_true_q_init,
                                   a=a, b=b, c=c, T_mode=T_mode) for i in range(n_sim)]
    results = ray.get(futures)
    return package_results(results)


@ray.remote
def train_history_remote(seed, dist_type, taus, client_rs, n_clients, T, E_typ,
                 E_cons, gene_process, mode, use_true_q_init=False, 
                 a=0.51, b=100, c=2, return_history=True,T_mode='rounds'):
    return train(seed, dist_type, taus, client_rs, n_clients, T, E_typ,
                 E_cons, gene_process, mode, use_true_q_init=use_true_q_init,
                 a=a, b=b, c=c, return_history=return_history, T_mode=T_mode)

def run_federated_trajectory(dist_type, taus, client_rs, n_clients, 
                            T, E_typ, E_cons, gene_process, mode, use_true_q_init=False, base_seed=2025,
                            a=0.51, b=100, c=2,T_mode='rounds'):
    """Run a single federated experiment with return_history=True to return the training history."""

    future = train_history_remote.remote(base_seed, dist_type, taus, client_rs, n_clients, T, E_typ,
                E_cons, gene_process, mode, use_true_q_init=use_true_q_init,
                a=a, b=b, c=c, return_history=True, T_mode=T_mode)
    result = ray.get(future)
    return result