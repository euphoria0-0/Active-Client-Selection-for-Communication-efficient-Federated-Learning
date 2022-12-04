from .trainer import Trainer
import numpy as np
import torch.nn.functional as F
import torch
from copy import deepcopy


class Client(object):
    def __init__(self, client_idx, nTrain, local_train_data, local_test_data, model, args):
        """
        A client
        ---
        Args
            client_idx: index of the client
            nTrain: number of train dataset of the client
            local_train_data: train dataset of the client
            local_test_data: test dataset of the client
            model: given model for the client
            args: arguments for overall FL training
        """
        self.client_idx = client_idx
        self.test_data = local_test_data
        self.device = args.device
        self.trainer = Trainer(model, args)
        self.num_epoch = args.num_epoch  # E: number of local epoch
        self.nTrain = nTrain
        self.loss_div_sqrt = args.loss_div_sqrt
        self.loss_sum = args.loss_sum

        self.labeled_indices = [*range(nTrain)]
        self.labeled_data = local_train_data  # train_data


    def train(self, global_model):
        """
        train each client
        ---
        Args
            global_model: given current global model
        Return
            result = model, loss, acc
        """
        # SET MODEL
        self.trainer.set_model(global_model)

        # TRAIN
        if self.num_epoch == 0:  # no SGD updates
            result = self.trainer.train_E0(self.labeled_data)
        else:
            result = self.trainer.train(self.labeled_data)
        #result['model'] = self.trainer.get_model()

        # total loss / sqrt (# of local data)
        if self.loss_div_sqrt:  # total loss / sqrt (# of local data)
            result['metric'] *= np.sqrt(len(self.labeled_data))  # loss * n_k / np.sqrt(n_k)
        elif self.loss_sum:
            result['metric'] *= len(self.labeled_data)  # total loss
        
        return result

    def test(self, model, test_on_training_data=False):
        # TEST
        if test_on_training_data:
            # test on training dataset
            result = self.trainer.test(model, self.labeled_data)
        else:
            # test on test dataset
            result = self.trainer.test(model, self.test_data)
        return result

    def get_client_idx(self):
        return self.client_idx