'''
Reference:
    https://github.com/Accenture/Labs-Federated-Learning/tree/clustered_sampling
'''
import torchvision.datasets as D
import torchvision.transforms as T
from torch.utils.data import TensorDataset
import torch
import numpy as np
import os



class PartitionedCIFAR10Dataset(object):
    def __init__(self, data_dir, args):
        '''
        partitioned CIFAR10 datatset according to a Dirichlet distribution
        '''
        self.num_classes = 10
        self.train_num_clients = 100 if args.total_num_clients is None else args.total_num_clients
        self.test_num_clients = 100 if args.total_num_clients is None else args.total_num_clients
        self.balanced = False
        self.alpha = args.dirichlet_alpha

        self._init_data(data_dir)
        print(f'Total number of users: {self.train_num_clients}')
    
    def _init_data(self, data_dir):
        file_name = os.path.join(data_dir, 'PartitionedCIFAR10_preprocessed_.pickle')
        if os.path.isfile(file_name):
            with open(file_name, 'rb') as f:
                dataset = pickle.load(f)
        else:
            matrix = np.random.dirichlet([self.alpha] * self.num_classes, size=self.train_num_clients)

            train_data = self.partition_CIFAR_dataset(data_dir, matrix, train=True)
            test_data = self.partition_CIFAR_dataset(data_dir, matrix, train=False)

            dataset = {
                'train': train_data, 
                'test' : test_data
            }
            

        self.dataset = dataset


    def partition_CIFAR_dataset(self, data_dir, matrix, train):
        """Partition dataset into `n_clients`.
        Each client i has matrix[k, i] of data of class k"""

        transform = [
            T.ToTensor(),
            T.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
        ]
        dataset = D.CIFAR10(data_dir, train=train, download=True, transform=transform)
        
        n_clients = self.train_num_clients if train else self.test_num_clients

        list_clients_X = [[] for i in range(n_clients)]
        list_clients_y = [[] for i in range(n_clients)]

        if self.balanced:
            n_samples = [500] * n_clients
        elif not self.balanced and train:
            n_samples = [100] * 10 + [250] * 30 + [500] * 30 + [750] * 20 + [1000] * 10
        elif not self.balanced and not train:
            n_samples = [20] * 10 + [50] * 30 + [100] * 30 + [150] * 20 + [200] * 10
        
        # custom
        # if train:
        #     n_samples = [5] * 10 + [10] * 20 + [100] * 10 + [250] * 29 + [500] * 30 + [750] * 20 + [1000] * 10
        # else:
        #     n_samples = [10] * 29 + [20] * 10 + [50] * 30 + [100] * 30 + [150] * 20 + [200] * 10

        list_idx = []
        for k in range(self.num_classes):

            idx_k = np.where(np.array(dataset.targets) == k)[0]
            list_idx += [idx_k]

        for idx_client, n_sample in enumerate(n_samples):

            clients_idx_i = []
            client_samples = 0

            for k in range(self.num_classes):

                if k < self.num_classes:
                    samples_digit = int(matrix[idx_client, k] * n_sample)
                if k == self.num_classes:
                    samples_digit = n_sample - client_samples
                client_samples += samples_digit

                clients_idx_i = np.concatenate(
                    (clients_idx_i, np.random.choice(list_idx[k], samples_digit))
                )

            clients_idx_i = clients_idx_i.astype(int)

            for idx_sample in clients_idx_i:

                list_clients_X[idx_client] += [dataset.data[idx_sample]]
                list_clients_y[idx_client] += [dataset.targets[idx_sample]]

            list_clients_X[idx_client] = np.array(list_clients_X[idx_client])

        return {
            'data': {idx: TensorDataset(torch.Tensor(list_clients_X[idx]).permute(0, 3, 1, 2), torch.tensor(list_clients_y[idx])) for idx in range(len(list_clients_X))}, # (list_clients_X, list_clients_y),
            'data_sizes': {idx: len(list_clients_y[idx]) for idx in range(len(list_clients_X))}
        }