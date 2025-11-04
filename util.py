import numpy as np
from scipy.stats import norm, cauchy, laplace
import ray


def generate_data(dist_type, tau, n_samples, mu=0):
    """Generate a data stream from a specified distribution with location mu, and return the data along with the true quantiles."""
    if dist_type == 'normal':
        data = np.random.normal(mu, 1, n_samples)      # Normal(mu, 1)
        true_q = mu + norm.ppf(tau)
    elif dist_type == 'uniform':
        low, high = mu - 1, mu + 1
        data = np.random.uniform(low, high, n_samples) # Uniform(mu-1, mu+1)
        true_q = low + (high - low) * tau
    elif dist_type == 'cauchy':
        data = np.random.standard_cauchy(n_samples) + mu  # Cauchy(mu, 1)
        true_q = mu + cauchy.ppf(tau)
    elif dist_type == 'laplace':
        data = np.random.laplace(mu, 1, n_samples)       # Laplace(mu, 1)
        true_q = mu + laplace.ppf(tau)
    
    else:
        raise ValueError("Unsupported distribution type. Please choose 'normal', 'uniform', 'cauchy', or 'laplace'")
    
    return data, true_q

def distribute_data(data, n_clients):
    """Randomly shuffle data and distribute proportionally to clients"""
    data = np.random.permutation(data)
    return np.split(data, n_clients)

def package_results(raw_results):
    """Standardized format for result packaging"""
    true_q, estimates, variances, maes = zip(*raw_results)
    return {
        'estimates': np.array(estimates),
        'variances': np.array(variances),
        'maes': np.array(maes),
        'true_q': np.array(true_q)
    }

def analyze_results(results, z_score=6.74735):
    """Analyze simulation results"""
    est = results['estimates']
    var = results['variances']
    true_q = results['true_q']
    
    # Calculate coverage probability
    lower = est - z_score * np.sqrt(var)
    upper = est + z_score * np.sqrt(var)
    coverage = np.mean((true_q >= lower) & (true_q <= upper))
    
    return {
        'coverage': coverage,
        'mae': np.mean(np.abs(est-true_q))
    }

