from copy import deepcopy
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score


class Trainer:
    def __init__(self, model, args):
        """
        trainer
        ---
        Args
            model: given model  for training (or test)
            args: arguments for FL training
        """
        self.device = args.device
        self.num_classes = args.num_classes

        # hyperparameter
        self.lr = args.lr_local
        self.wdecay = args.wdecay
        self.momentum = args.momentum
        self.num_epoch = args.num_epoch    # num of local epoch E
        self.num_updates = args.num_updates  # num of local updates u
        self.batch_size = args.batch_size  # local batch size B
        self.loader_kwargs = {'batch_size': self.batch_size, 'pin_memory': True, 'shuffle': True}

        # model
        self.model = model
        self.client_optimizer = args.client_optimizer


    def get_model(self):
        """
        get current model
        """
        self.model.eval()
        return self.model

    def set_model(self, model):
        """
        set current model for training
        """
        self.model.load_state_dict(model.cpu().state_dict())

    def train(self, data):
        """
        train
        ---
        Args
            data: dataset for training
        Returns
            accuracy, loss
        """
        dataloader = DataLoader(data, **self.loader_kwargs)

        self.model = self.model.to(self.device)

        
        self.model.train()

        # optimizer
        if self.client_optimizer == 'sgd':
            optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.wdecay)
        else:
            optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.wdecay)

        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(self.num_epoch):
            loss_lst = []
            output_lst, res_lst = torch.empty((0, self.num_classes)).to(self.device), torch.empty((0, self.num_classes)).to(self.device)
            min_loss, num_ot = np.inf, 0
            train_loss, correct, total = 0., 0, 0
            probs = 0
            for num_update, (input, labels) in enumerate(dataloader):
                input, labels = input.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                output = self.model(input)
                _, preds = torch.max(output.detach().data, 1)

                loss = criterion(output, labels.long())

                loss.backward()
                optimizer.step()

                train_loss += loss.detach().item() * input.size(0)
                
                correct += preds.eq(labels).sum().cpu().data.numpy()
                total += input.size(0)

                
                if self.num_updates is not None and num_update + 1 == self.num_updates:
                    if total < self.batch_size:
                        print(f'break! {total}', end=' ')
                    break

                del input, labels, output
                

        self.model = self.model.cpu()

        assert total > 0
            
        result = {'loss': train_loss / total, 'acc': correct / total, 'metric': train_loss / total}
        
        
        # if you track each client's loss
        # sys.stdout.write(r'\nLoss {:.6f} Acc {:.4f}'.format(result['loss'], result['acc']))
        # sys.stdout.flush()

        return result

    def train_E0(self, data):
        """
        train with no local SGD updates
        ---
        Args
            data: dataset for training
        Returns
            accuracy, loss
        """
        dataloader = DataLoader(data, **self.loader_kwargs)

        self.model = self.model.to(self.device)
        self.model.train()

        # optimizer
        if self.client_optimizer == 'sgd':
            optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum,
                                       weight_decay=self.wdecay)
        else:
            optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.wdecay)

        criterion = nn.CrossEntropyLoss()

        correct, total = 0, 0
        batch_loss = []
        for input, labels in dataloader:
            input, labels = input.to(self.device), labels.to(self.device)
            optimizer.zero_grad()
            output = self.model(input)

            loss = criterion(output, labels.long())
            _, preds = torch.max(output.data, 1)

            batch_loss.append(loss * input.size(0))  ##### loss sum
            total += input.size(0).detach().cpu().data.numpy()
            correct += preds.eq(labels).sum().detach().cpu().data.numpy()

        train_acc = correct / total
        avg_loss = sum(batch_loss) / total

        avg_loss.backward()
        optimizer.step()

        sys.stdout.write('\rTrainLoss {:.6f} TrainAcc {:.4f}'.format(avg_loss, train_acc))

        result = {'loss': avg_loss.detach().cpu(), 'acc': train_acc}

        return result


    #@torch.no_grad()
    def test(self, model, data, ema=False):
        """
        test
        ---
        Args
            model: model for test
            data: dataset for test
        Returns
            accuracy, loss, AUC (optional)
        """
        dataloader = DataLoader(data, **self.loader_kwargs)

        model = model.to(self.device)
        model.eval()

        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            test_loss, correct, total = 0., 0, 0
            y_true, y_score = np.empty((0)), np.empty((0))
            output_lst, res_lst = torch.empty((0, self.num_classes)), torch.empty((0, self.num_classes))

            for input, labels in dataloader:
                input, labels = input.to(self.device), labels.to(self.device)
                output = model(input)

                loss = criterion(output, labels.long())
                _, preds = torch.max(output.data, 1)

                test_loss += loss.detach().cpu().item() * input.size(0)
                correct += preds.eq(labels).sum().detach().cpu().data.numpy()
                total += input.size(0)

                if self.num_classes == 2:
                    y_true = np.append(y_true, labels.detach().cpu().numpy(), axis=0)
                    y_score = np.append(y_score, preds.detach().cpu().numpy(), axis=0)
                
                del input, labels, output, preds

        assert total > 0

        result = {'loss': test_loss / total, 'acc': correct / total}

        #if self.num_classes == 2:
        #    auc = roc_auc_score(y_true, y_score)
        #    result['auc'] = auc

        return result