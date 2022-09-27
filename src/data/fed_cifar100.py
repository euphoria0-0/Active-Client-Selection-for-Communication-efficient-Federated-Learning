'''
Reference:
    FedML: https://github.com/FedML-AI/FedML
'''
import os
import sys

import h5py
import pickle
import numpy as np
import torch
from torch.utils.data import TensorDataset
import torchvision.transforms as T


class FederatedCIFAR100Dataset:
    def __init__(self, data_dir, args):
        self.num_classes = 100
        self.train_num_clients = 500
        self.test_num_clients = 100
        self.batch_size = args.batch_size # local batch size for local training # 20

        self._init_data(data_dir)
        print(f'Total number of users: {self.train_num_clients}')

    def _init_data(self, data_dir):
        file_name = os.path.join(data_dir, 'FedCIFAR100_preprocessed.pickle')
        if os.path.isfile(file_name):
            with open(file_name, 'rb') as f:
                dataset = pickle.load(f)
        else:
            dataset = preprocess(data_dir, self.train_num_clients)
        self.dataset = dataset


def preprocess(data_dir, num_clients=None):
    train_data = h5py.File(os.path.join(data_dir, 'fed_cifar100_train.h5'), 'r')
    test_data = h5py.File(os.path.join(data_dir, 'fed_cifar100_test.h5'), 'r')

    train_ids = list(train_data['examples'].keys())
    test_ids = list(test_data['examples'].keys())
    num_clients_train = len(train_ids)
    num_clients_test = len(test_ids)
    print(f'num_clients_train {num_clients_train} num_clients_test {num_clients_test}')

    # local dataset
    train_data_local_dict, train_data_local_num_dict = {}, {}
    test_data_local_dict, test_data_local_num_dict = {}, {}

    # train
    for client_idx in range(num_clients_train):
        client_id = train_ids[client_idx]

        train_x = np.expand_dims(train_data['examples'][client_id]['image'][()], axis=1)
        train_y = train_data['examples'][client_id]['label'][()]

        # preprocess
        train_x = preprocess_cifar_img(torch.tensor(train_x), train=True)

        local_data = TensorDataset(torch.Tensor(train_x), torch.Tensor(train_y))
        train_data_local_dict[client_idx] = local_data
        train_data_local_num_dict[client_idx] = len(train_x)

    # test
    for client_idx in range(num_clients_test):
        client_id = test_ids[client_idx]

        test_x = np.expand_dims(test_data['examples'][client_id]['image'][()], axis=1)
        test_y = test_data['examples'][client_id]['label'][()]

        # preprocess
        test_x = preprocess_cifar_img(torch.tensor(test_x), train=False)

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

    with open(os.path.join(data_dir, 'FedCIFAR100_preprocessed.pickle'), 'wb') as f:
        pickle.dump(dataset, f)

    return dataset





def cifar100_transform(img_mean, img_std, train = True, crop_size = (24,24)):
    """cropping, flipping, and normalizing."""
    if train:
        return T.Compose([
            T.ToPILImage(),
            T.RandomCrop(crop_size),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=img_mean, std=img_std),
        ])
    else:
        return T.Compose([
            T.ToPILImage(),
            T.CenterCrop(crop_size),
            T.ToTensor(),
            T.Normalize(mean=img_mean, std=img_std),
        ])


def preprocess_cifar_img(img, train):
    # scale img to range [0,1] to fit ToTensor api
    img = torch.div(img, 255.0)
    transoformed_img = torch.stack(
        [cifar100_transform(i.type(torch.DoubleTensor).mean(),
                            i.type(torch.DoubleTensor).std(),
                            train)
         (i[0].permute(2,0,1)) ##
         for i in img])
    return transoformed_img
