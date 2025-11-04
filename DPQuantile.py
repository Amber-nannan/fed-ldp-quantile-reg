import numpy as np
from typing import Optional

class DPQuantile:
    """Base class for differential privacy quantile estimation"""
        
    def __init__(self, tau=0.5, r=0.5, true_q=None,
     track_history=False, burn_in_ratio=0, use_true_q_init=False,a=0.51,b=100,c=2,
                seed=2025):
        self.tau = tau
        self.r = r
        self.true_q = true_q
        self.track_history = track_history
        self.burn_in_ratio = burn_in_ratio
        self.use_true_q_init = use_true_q_init 
        self.q_avg_history = {}
        self.variance_history = {}
        self.a = a
        self.b = b
        self.c0 = c
        self.seed=seed

    def _lr_schedule(self,step,c0=2,a=0.51,b=100):
        """
        Learning rate schedule
        """
        lr = c0 / (step**a + b)
        return lr

    def reset(self, q_est: Optional[float]=None):
        """Reset training state"""
        if self.use_true_q_init and self.true_q is not None:
            self.q_est = self.true_q  # Start from true value
        elif q_est:
            self.q_est = q_est
        else:
            np.random.seed(self.seed)
            self.q_est = np.random.normal(0,1)
            
        self.Q_avg = 0.0
        self.n = 0
        self.step = 0
        
        # Online inference statistics
        self.v_a = 0.0
        self.v_b = 0.0
        self.v_s = 0.0
        self.v_q = 0.0
        self.errors = []

    def _compute_gradient(self, x):
        """Core gradient computation"""
        if np.random.rand() < self.r:
            s = int(x > self.q_est)
        else:
            s = np.random.binomial(1, 0.5)
        
        delta = ((1 - self.r + 2*self.tau*self.r)/2 if s 
                else -(1 + self.r - 2*self.tau*self.r)/2)
        return delta

    def _update_estimator(self, delta, lr):
        """Update parameter"""
        self.q_est += lr * delta
        self.step += 1

    def _update_stats(self):
        """Update statistics"""
        self.n += 1
        prev_weight = (self.n - 1) / self.n
        self.Q_avg = prev_weight * self.Q_avg + self.q_est / self.n
        
        # Update variance statistics
        term = self.n**2
        self.v_a += term * self.Q_avg**2
        self.v_b += term * self.Q_avg
        self.v_q += term
        self.v_s += 1

        # Record Q_avg and variance for the current sample size
        self.q_avg_history[self.n] = self.Q_avg
        self.variance_history[self.n] = self.get_variance()
        
        if self.track_history and self.true_q is not None:
            self.errors.append(np.abs(self.Q_avg - self.true_q))

    def fit(self, data_stream):
        """Single-machine training method"""
        self.reset()
        n_samples = len(data_stream)
        burn_in = int(n_samples * self.burn_in_ratio)  # Calculate burn-in sample size
        for idx, x in enumerate(data_stream):
            # Calculate learning rate for the current step
            lr = self._lr_schedule(self.step + 1,
                            c0=self.c0,a=self.a,b=self.b)
            
            # Compute gradient and update estimator
            delta = self._compute_gradient(x)
            self._update_estimator(delta, lr)
            
            # Skip statistics update during burn-in phase
            if idx >= burn_in:
                self._update_stats()
            
            # Early stopping check
            if self.step >= n_samples:
                break

    def get_stats_history(self):
        """Get the history of Q_avg and variance statistics"""
        stats = {
            "q_avg": self.q_avg_history,
            "variance": self.variance_history
        }
        return stats

    def get_variance(self):
        """Get variance estimation"""
        if self.n == 0:
            return 0.0
        return (self.v_a - 2*self.Q_avg*self.v_b + 
               (self.Q_avg**2)*self.v_q) / (self.n**2 * self.v_s)
