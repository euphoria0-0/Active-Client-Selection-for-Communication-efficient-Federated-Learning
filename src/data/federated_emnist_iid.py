import os
import h5py
import pickle
import numpy as np
import torch
from torch.utils.data import TensorDataset


class FederatedEMNISTDatasetIID:
    def __init__(self, data_dir, args):
        self.num_classes = 62
        # self.train_num_clients = 3400 if args.total_num_clients is None else args.total_num_clients
        # self.test_num_clients = 3400 if args.total_num_clients is None else args.total_num_clients
        self.min_num_samples = 100
        # min_num_samples = 150; num_clients = 2492
        # min_num_samples = 100; num_clients = 

        self._init_data(data_dir)
        self.train_num_clients = len(self.dataset['train']['data_sizes'].keys())
        self.test_num_clients = len(self.dataset['test']['data_sizes'].keys())
        print(f'#TrainClients {self.train_num_clients} #TestClients {self.test_num_clients}')

    def _init_data(self, data_dir):
        file_name = os.path.join(data_dir, 'FederatedEMNIST_preprocessed_IID.pickle')
        if os.path.isfile(file_name):
            print('> read dataset ...')
            with open(file_name, 'rb') as f:
                dataset = pickle.load(f)
        else:
            print('> create dataset ...')
            dataset = preprocess(data_dir, self.min_num_samples)
            # with open(file_name, 'wb') as f:
            #     pickle.dump(dataset, f)
        self.dataset = dataset


def preprocess(data_dir, min_num_samples):
    train_data = h5py.File(os.path.join(data_dir, 'fed_emnist_train.h5'), 'r')
    test_data = h5py.File(os.path.join(data_dir, 'fed_emnist_test.h5'), 'r')

    train_ids = list(train_data['examples'].keys())
    test_ids = list(test_data['examples'].keys())
    num_clients_train = len(train_ids)
    num_clients_test = len(test_ids)
    print(f'#TrainClients {num_clients_train} #TestClients {num_clients_test}')

    # local dataset
    train_data_local_dict, train_data_local_num_dict = {}, {}
    test_data_local_dict, test_data_local_num_dict = {}, {}

    idx = 0

    for client_idx in range(num_clients_train):
        client_id = train_ids[client_idx]

        # train
        train_x = np.expand_dims(train_data['examples'][client_id]['pixels'][()], axis=1)
        train_y = train_data['examples'][client_id]['label'][()]

        if len(train_x) < min_num_samples:
            continue
        train_x = train_x[:min_num_samples]
        train_y = train_y[:min_num_samples]

        local_data = TensorDataset(torch.Tensor(train_x), torch.Tensor(train_y))
        train_data_local_dict[idx] = local_data
        train_data_local_num_dict[idx] = len(train_x)

        # test
        test_x = np.expand_dims(test_data['examples'][client_id]['pixels'][()], axis=1)
        test_y = test_data['examples'][client_id]['label'][()]
        local_data = TensorDataset(torch.Tensor(test_x), torch.Tensor(test_y))
        test_data_local_dict[idx] = local_data
        test_data_local_num_dict[idx] = len(test_x)
        if len(test_x) == 0:
            print(client_idx)
        idx += 1

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

    return dataset
