import numpy as np


class ClientSelection:
    def __init__(self, total, device):
        self.total = total
        self.device = device

    def select(self, n, client_idxs, metric):
        pass

    def save_selected_clients(self, client_idxs, results):
        tmp = np.zeros(self.total)
        tmp[client_idxs] = 1
        tmp.tofile(results, sep=',')
        results.write("\n")

    def save_results(self, arr, results, prefix=''):
        results.write(prefix)
        np.array(arr).astype(np.float32).tofile(results, sep=',')
        results.write("\n")



'''Random Selection'''
class RandomSelection(ClientSelection):
    def __init__(self, total, device):
        super().__init__(total, device)

    def select(self, n, client_idxs, metric=None):
        selected_client_idxs = np.random.choice(client_idxs, size=n, replace=False)
        return selected_client_idxs
