from copy import deepcopy
from collections import Counter
import numpy as np

from .client_selection import ClientSelection



'''Active Federated Learning'''
class ActiveFederatedLearning(ClientSelection):
    def __init__(self, total, device, args):
        super().__init__(total, device)
        self.alpha1 = args.alpha1 #0.75
        self.alpha2 = args.alpha2 #0.01
        self.alpha3 = args.alpha3 #0.1
        self.save_probs = args.save_probs

    def select(self, n, client_idxs, metric, round=0, results=None):
        # set sampling distribution
        values = np.exp(np.array(metric) * self.alpha2)
        # 1) select 75% of K(total) users
        num_drop = len(metric) - int(self.alpha1 * len(metric))
        drop_client_idxs = np.argsort(metric)[:num_drop]
        probs = deepcopy(values)
        probs[drop_client_idxs] = 0
        probs /= sum(probs)
        #probs = np.nan_to_num(probs, nan=max(probs))
        # 2) select 99% of m users using prob.
        num_select = int((1 - self.alpha3) * n)
        #np.random.seed(round)
        selected = np.random.choice(len(metric), num_select, p=probs, replace=False)
        # 3) select 1% of m users randomly
        not_selected = np.array(list(set(np.arange(len(metric))) - set(selected)))
        selected2 = np.random.choice(not_selected, n - num_select, replace=False)
        selected_client_idxs = np.append(selected, selected2, axis=0)
        print(f'{len(selected_client_idxs)} selected users: {selected_client_idxs}')

        if self.save_probs:
            self.save_results(metric, results, f'{round},loss,')
            self.save_results(values, results, f'{round},value,')
            self.save_results(probs, results, f'{round},prob,')
        return selected_client_idxs.astype(int)



'''Power-of-Choice'''
class PowerOfChoice(ClientSelection):
    def __init__(self, total, device, d):
        super().__init__(total, device)
        #self.d = d

    def setup(self, n_samples):
        client_ids = sorted(n_samples.keys())
        n_samples = np.array([n_samples[i] for i in client_ids])
        self.weights = n_samples / np.sum(n_samples)

    def select_candidates(self, client_idxs, d):
        # 1) sample the candidate client set
        weights = np.take(self.weights, client_idxs)
        candidate_clients = np.random.choice(client_idxs, d, p=weights/sum(weights), replace=False)
        return candidate_clients

    def select(self, n, client_idxs, metric, round=0, results=None):
        # 3) select highest loss clients
        selected_client_idxs = np.argsort(metric)[-n:]
        return selected_client_idxs
