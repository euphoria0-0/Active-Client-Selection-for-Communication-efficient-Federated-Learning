'''
Client Selection for Federated Learning
'''
import os
import sys
import time

AVAILABLE_WANDB = True
try:
    import wandb
except ModuleNotFoundError:
    AVAILABLE_WANDB = False

import torch
import random

from .data import *
from .model import *
from .FL_core.server import Server
from .FL_core.client_selection import *
from .FL_core.federated_algorithm import *
from .utils import utils
from .utils.argparse import get_args



def load_data(args):
    if args.dataset == 'Reddit':
        return RedditDataset(args.data_dir, args)
    elif args.dataset == 'FederatedEMNIST':
        return FederatedEMNISTDataset(args.data_dir, args)
    elif args.dataset == 'FederatedEMNIST_IID':
        return FederatedEMNISTDatasetIID(args.data_dir, args)
    elif args.dataset == 'FederatedEMNIST_nonIID':
        return FederatedEMNISTDataset_nonIID(args.data_dir, args)
    elif args.dataset == 'FedCIFAR100':
        return FederatedCIFAR100Dataset(args.data_dir, args)
    elif args.dataset == 'CelebA':
        return CelebADataset(args.data_dir, args)
    elif args.dataset == 'PartitionedCIFAR10':
        return PartitionedCIFAR10Dataset(args.data_dir, args)


def create_model(args):
    if args.dataset == 'Reddit' and args.model == 'BLSTM':
        model = BLSTM(vocab_size=args.maxlen, num_classes=args.num_classes)
    elif args.dataset == 'FederatedEMNIST_nonIID' and args.model == 'CNN':
        model = CNN_DropOut(True)
    elif args.dataset == 'FederatedEMNIST_nonIID' and args.model == 'CNN':
        model = CNN_DropOut(True)
    elif 'FederatedEMNIST' in args.dataset and args.model == 'CNN':
        model = CNN_DropOut(False)
    elif args.dataset == 'FedCIFAR100' and args.model == 'ResNet':
        model = resnet18(num_classes=args.num_classes, group_norm=args.num_gn)  # ResNet18+GN
    elif args.dataset == 'CelebA' and args.model == 'CNN':
        model = ModelCNNCeleba()
    elif args.dataset == 'PartitionedCIFAR10':
        model = CNN_CIFAR_dropout()

    model = model.to(args.device)
    if args.parallel:
        model = torch.nn.DataParallel(model, output_device=0)
    return model


def federated_algorithm(dataset, model, args):
    train_sizes = dataset['train']['data_sizes']
    if args.fed_algo == 'FedAdam':
        return FedAdam(train_sizes, model, args=args)
    else:
        return FedAvg(train_sizes, model)


def client_selection_method(args):
    #total = args.total_num_client if args.num_available is None else args.num_available
    kwargs = {'total': args.total_num_client, 'device': args.device}
    if args.method == 'Random':
        return RandomSelection(**kwargs)
    elif args.method == 'AFL':
        return ActiveFederatedLearning(**kwargs, args=args)
    elif args.method == 'Cluster1':
        return ClusteredSampling1(**kwargs, n_cluster=args.num_clients_per_round)
    elif args.method == 'Cluster2':
        return ClusteredSampling2(**kwargs, dist=args.distance_type)
    elif args.method == 'Pow-d':
        assert args.num_candidates is not None
        return PowerOfChoice(**kwargs, d=args.num_candidates)
    elif args.method == 'DivFL':
        assert args.subset_ratio is not None
        return DivFL(**kwargs, subset_ratio=args.subset_ratio)
    elif args.method == 'GradNorm':
        return GradNorm(**kwargs)
    else:
        raise('CHECK THE NAME OF YOUR SELECTION METHOD')



if __name__ == '__main__':
    # set up
    args = get_args()
    if args.comment != '': args.comment = '-'+args.comment
    #if args.labeled_ratio < 1: args.comment = f'-L{args.labeled_ratio}{args.comment}'
    if args.fed_algo != 'FedAvg': args.comment = f'-{args.fed_algo}{args.comment}'
    
    # save to wandb
    args.wandb = AVAILABLE_WANDB
    if args.wandb:
        wandb.init(
            project=f'AFL-{args.dataset}-{args.num_clients_per_round}-{args.num_available}-{args.total_num_clients}',
            name=f"{args.method}{args.comment}",
            config=args,
            dir='.',
            save_code=True
        )
        wandb.run.log_code(".", include_fn=lambda x: 'src/' in x or 'main.py' in x)

    # fix seed
    if args.fix_seed:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # device setting
    if args.gpu_id == 'cpu' or not torch.cuda.is_available():
        args.device = 'cpu'
    else:
        if ',' in args.gpu_id:
            os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_id
        args.device = torch.device(f"cuda:{args.gpu_id[0]}")
        torch.cuda.set_device(args.device)
        print('Current cuda device ', torch.cuda.current_device())

    # set data
    data = load_data(args)
    args.num_classes = data.num_classes
    args.total_num_client, args.test_num_clients = data.train_num_clients, data.test_num_clients
    dataset = data.dataset

    # set model
    model = create_model(args)
    client_selection = client_selection_method(args)
    fed_algo = federated_algorithm(dataset, model, args)

    # save results
    files = utils.save_files(args)

    ## train
    # set federated optim algorithm
    ServerExecute = Server(dataset, model, args, client_selection, fed_algo, files)
    ServerExecute.train()
