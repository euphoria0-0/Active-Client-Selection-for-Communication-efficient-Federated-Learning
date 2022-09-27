'''
Reference:
    FedML: https://github.com/FedML-AI/FedML
'''
import os
import h5py
import pickle
import numpy as np
import torch
from torch.utils.data import TensorDataset


class FederatedEMNISTDataset:
    def __init__(self, data_dir, args):
        self.num_classes = 62
        self.train_num_clients = 3400 if args.total_num_clients is None else args.total_num_clients
        self.test_num_clients = 3400 if args.total_num_clients is None else args.total_num_clients
        self.batch_size = args.batch_size # local batch size for local training

        self._init_data(data_dir)
        print(f'Total number of users: {self.train_num_clients}')

    def _init_data(self, data_dir):
        file_name = os.path.join(data_dir, 'FederatedEMNIST_preprocessed_.pickle')
        if os.path.isfile(file_name):
            with open(file_name, 'rb') as f:
                dataset = pickle.load(f)
        else:
            dataset = preprocess(data_dir, self.train_num_clients)
            #dataset = batch_preprocess(data_dir, self.batch_size, self.train_num_clients)
        self.dataset = dataset


def preprocess(data_dir, num_clients=None):
    train_data = h5py.File(os.path.join(data_dir, 'fed_emnist_train.h5'), 'r')
    test_data = h5py.File(os.path.join(data_dir, 'fed_emnist_test.h5'), 'r')

    train_ids = list(train_data['examples'].keys())
    test_ids = list(test_data['examples'].keys())
    num_clients_train = len(train_ids) if num_clients is None else num_clients
    num_clients_test = len(test_ids) if num_clients is None else num_clients
    print(f'num_clients_train {num_clients_train} num_clients_test {num_clients_test}')

    # local dataset
    train_data_local_dict, train_data_local_num_dict = {}, {}
    test_data_local_dict, test_data_local_num_dict = {}, {}

    for client_idx in range(num_clients_train):
    #for client_idx, x in enumerate([1528, 628, 1526, 1766, 2515, 3050, 1104, 447, 67, 178]):
        client_id = train_ids[client_idx]
        #client_id = train_ids[x]

        # train
        train_x = np.expand_dims(train_data['examples'][client_id]['pixels'][()], axis=1)
        train_y = train_data['examples'][client_id]['label'][()]
        local_data = TensorDataset(torch.Tensor(train_x), torch.Tensor(train_y))
        train_data_local_dict[client_idx] = local_data
        train_data_local_num_dict[client_idx] = len(train_x)

        # test
        test_x = np.expand_dims(test_data['examples'][client_id]['pixels'][()], axis=1)
        test_y = test_data['examples'][client_id]['label'][()]
        local_data = TensorDataset(torch.Tensor(test_x), torch.Tensor(test_y))
        test_data_local_dict[client_idx] = local_data
        test_data_local_num_dict[client_idx] = len(test_x)
        if len(test_x) == 0:
            print(client_idx)

    train_data.close()
    test_data.close()

    dataset = {}
    dataset['train'] = {
        'data_sizes': train_data_local_num_dict,
        'data': train_data_local_dict,
    }
    dataset['test'] = {
        'data_sizes': test_data_local_num_dict,
        'data': test_data_local_dict,
    }

    with open(os.path.join(data_dir, 'FederatedEMNIST_preprocessed.pickle'), 'wb') as f:
        pickle.dump(dataset, f)

    return dataset