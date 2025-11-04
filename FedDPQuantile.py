import numpy as np
from DPQuantile import DPQuantile

class FedDPQuantile(DPQuantile):
    """Federated Differential Privacy Quantile Estimation"""
    
    def __init__(self, n_clients=1, pk=None, client_rs=None,
                 taus=0.5, **kwargs):
        super().__init__(**kwargs)
        self.n_clients = n_clients
        self.pk_init = pk
        self.global_q_avg_history = {}   
        self.global_variance_history = {}
        
        # Handle clients' privacy parameters
        self.taus = [taus] * n_clients if isinstance(taus, (int, float)) else taus
        self.client_rs = ([self.r] * n_clients if client_rs is None
                     else [self.r] * n_clients if isinstance(client_rs, (int, float))
                     else client_rs)
        
        if len(self.client_rs) != n_clients:
            raise ValueError("Length of client_rs must match n_clients")
        self.clients = [self._create_client(i) for i in range(self.n_clients)]
        
        np.random.seed(self.seed)
        q_est = np.random.normal(0, 1)  # Random initial value
        for clients in self.clients:
            clients.reset(q_est)    # Same random initial value for each machine
        
            
    def _lr_schedule(self,step,c0=2,a=0.51,b=100):
        """
        Learning rate schedule
        """
        lr = c0 / (step**a + b)  # lr = c0 / (step**a + 500)
        return lr

    def _create_client(self, client_idx):
        """Create client instance (with independent privacy parameters)"""
        tau_avg = np.mean(self.taus)
        return DPQuantile(tau=tau_avg, r=self.client_rs[client_idx],
                          true_q=self.true_q,
                         use_true_q_init=self.use_true_q_init)

    def _get_batch(self, client_idx,Em=1):
        """Get client data batch"""
        try:
            return [next(self.data_streams[client_idx]) for _ in range(Em)]
        except StopIteration:
            return None

    def _aggregate(self, params):
        """Parameter aggregation (default FedAvg)"""
        return np.average(params, weights=self.pk)

    def fit(self,clients_data,Em_list,warm_up=0.05):
        self.Em_list = Em_list
        self.reset()
        self.data_streams = [iter(data) for data in clients_data]
        self.pk = self.pk_init if self.pk_init else np.ones(self.n_clients)/self.n_clients

        """Federated training main loop"""
        M = len(self.Em_list)
        warm_up_em = int(np.sum(Em_list) * warm_up)
        for m in range(M):
            m_prime = m - warm_up_em if m > warm_up_em else 0
            Em = self.Em_list[m]
            # Parallel local updates
            client_params = []
            for c in range(self.n_clients):
                batch = self._get_batch(c,Em)
                if batch is None:  # Data exhausted
                    return 0
                
                client = self.clients[c]
                for x in batch:
                    delta = client._compute_gradient(x)
                    lr = self._lr_schedule(m_prime + 1,c0=self.c0,a=self.a,b=self.b) 
                    client._update_estimator(delta, lr/Em)
                    client._update_stats()
                
                client_params.append(client.q_est)

            # Global aggregation and update
            global_est = self._aggregate(client_params)
            self._sync_global_state(global_est)

    def _sync_global_state(self, global_est):
        """Synchronize global state"""
        # Update server state
        self.n += 1
        prev_weight = (self.n - 1) / self.n
        self.Q_avg = prev_weight * self.Q_avg + global_est / self.n
        
        # Update variance statistics
        Em = self.Em_list[self.n-1]
        term = self.n**2 / Em
        self.v_a += term * self.Q_avg**2
        self.v_b += term * self.Q_avg
        self.v_s += 1 / Em
        self.v_q += term
        
        # Record global Q_avg and variance statistics (using clients[0].n as index)
        local_n = self.clients[0].n
        self.global_q_avg_history[local_n] = self.Q_avg
        self.global_variance_history[local_n] = self.get_variance()
        
        # Synchronize to all clients
        for client in self.clients:
            client.q_est = global_est
            client.Q_avg = self.Q_avg
    
    def get_stats_history(self):
        """Get the history of Q_avg and variance statistics"""
        # Return local client Q_avg history
        local_stats = {f"client_{i}": client.q_avg_history for i, client in enumerate(self.clients)}
        
        # Return global Q_avg and variance history
        global_stats = {
            "global_q_avg": self.global_q_avg_history,
            "global_variance": self.global_variance_history
        }
        
        return {
            "local": local_stats,
            "global": global_stats
        }