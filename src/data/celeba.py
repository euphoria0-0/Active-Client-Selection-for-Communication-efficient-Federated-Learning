from PIL import Image
import os
import pickle
from tqdm import tqdm
from torch.utils.data import TensorDataset
from torch import Tensor
import torch
import numpy as np
import json
from collections import defaultdict




class CelebADataset(object):
    def __init__(self, data_dir, args):
        self.num_classes = 2
        self.min_num_samples = args.min_num_samples
        self.max_num_clients = args.total_num_clients
        self.img_size = 84

        self._init_data(data_dir)
        print(f'Total number of users: train {self.train_num_clients} test {self.test_num_clients}')


    def _init_data(self, data_dir):
        # file_name = os.path.join(data_dir, f'CelebA_preprocessed.pickle')
        file_name = os.path.join(data_dir, f'CelebA.pickle')
        if os.path.isfile(file_name):
            print('> read data ...')
            with open(file_name, 'rb') as f:
                dataset = pickle.load(f)
        else:
            # dataset = preprocess(data_dir, self.img_size)
            dataset = preprocess_online_read(data_dir, self.img_size)
            with open(file_name, 'wb') as f:
                pickle.dump(dataset, f)
        
        self.dataset = dataset

        self.train_num_clients = len(dataset["train"]["data_sizes"])
        self.test_num_clients = len(dataset["test"]["data_sizes"]) 



def preprocess(data_dir, img_size=84):
    img_dir = os.path.join(data_dir,'raw/img_align_celeba')

    train_clients, train_groups, train_data = read_dir(os.path.join(data_dir, 'train'))
    test_clients, test_groups, test_data = read_dir(os.path.join(data_dir, 'test'))

    assert train_clients == test_clients
    assert train_groups == test_groups

    clients = sorted(map(int, train_clients))

    trainset_data, trainset_datasize = {}, {}
    testset_data, testset_datasize = {}, {}

    for idx in tqdm(range(len(clients)), desc='create dataset'):
        client_id = str(clients[idx])
        # train data
        train_x = [load_image(i, img_dir, img_size) for i in train_data[client_id]['x']]
        train_y = list(map(int, train_data[client_id]['y']))
        trainset_data[idx] = TensorDataset(Tensor(train_x), Tensor(train_y))
        trainset_datasize[idx] = len(train_y)

        # test data
        test_x = [load_image(i, img_dir, img_size) for i in test_data[client_id]['x']]
        test_y = list(map(int, test_data[client_id]['y']))
        testset_data[idx] = TensorDataset(Tensor(test_x), Tensor(test_y))
        testset_datasize[idx] = len(test_y)

    dataset = {
        'train': {'data': trainset_data, 'data_sizes': trainset_datasize}, 
        'test': {'data': testset_data, 'data_sizes': testset_datasize}
    }
    return dataset
        


def read_dir(data_dir):
    clients = []
    groups = []
    data = defaultdict(lambda : None)

    files = os.listdir(data_dir)
    files = [f for f in files if f.endswith('.json')]
    for f in files:
        file_path = os.path.join(data_dir,f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        clients.extend(cdata['users'])
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        data.update(cdata['user_data'])

    clients = list(sorted(data.keys()))
    return clients, groups, data



def load_image(img_name, img_dir, img_size):
    img = Image.open(os.path.join(img_dir, img_name))
    img = img.resize((img_size, img_size)).convert('RGB')
    return np.array(img).transpose(2,0,1)




class CelebA_ClientData(object):
    def __init__(self, img_dir, img_size, dataset):
        self.img_dir = img_dir
        self.img_size = img_size
        self.dataset = dataset
        self.num_data = len(self.dataset['y'])

    def __getitem__(self, index):
        img_name = self.dataset['x'][index]
        data = self.load_image(img_name)
        target = torch.tensor(self.dataset['y'][index], dtype=torch.long)
        return data, target

    def __len__(self):
        return self.num_data
    
    def load_image(self, img_name):
        img = Image.open(os.path.join(self.img_dir, img_name))
        img = img.resize((self.img_size, self.img_size)).convert('RGB')
        img = torch.tensor(np.array(img).transpose(2,0,1)).float()
        return img


def preprocess_online_read(data_dir, img_size=84):
    img_dir = os.path.join(data_dir,'raw/img_align_celeba')

    train_clients, train_groups, train_data = read_dir(os.path.join(data_dir, 'train'))
    test_clients, test_groups, test_data = read_dir(os.path.join(data_dir, 'test'))

    assert train_clients == test_clients
    assert train_groups == test_groups

    clients = sorted(map(int, train_clients))

    trainset_data, trainset_datasize = {}, {}
    testset_data, testset_datasize = {}, {}

    for idx in range(len(clients)):
        client_id = str(clients[idx])
        # train data
        client_data = CelebA_ClientData(img_dir, img_size, train_data[client_id])
        trainset_data[idx] = client_data
        trainset_datasize[idx] = client_data.num_data

        # test data
        client_data = CelebA_ClientData(img_dir, img_size, test_data[client_id])
        testset_data[idx] = client_data
        testset_datasize[idx] = client_data.num_data

    dataset = {
        'train': {'data': trainset_data, 'data_sizes': trainset_datasize}, 
        'test': {'data': testset_data, 'data_sizes': testset_datasize}
    }
    return dataset