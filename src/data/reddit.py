import os
import pickle

import bz2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


class RedditDataset:
    def __init__(self, data_dir, args):
        self.num_classes = 2
        #self.train_size = 124638 # messages
        #self.test_size = 15568 # messages
        self.train_num_clients = 7668 # 7527 (paper)
        self.test_num_clients = 2099
        self.batch_size = args.batch_size #128
        self.maxlen = args.maxlen #400

        self._init_data(data_dir)
        print(f'Total number of users: {self.train_num_clients}')

    def _init_data(self, data_dir):
        file_name = os.path.join(data_dir, 'Reddit_preprocessed_7668.pickle')
        if os.path.isfile(file_name) and self.batch_size == 128 and self.maxlen == 400:
            with open(file_name, 'rb') as f:
                dataset = pickle.load(f) # user_id, num_data, text, label
        else:
            dataset = preprocess(data_dir)
        self.dataset = dataset



def preprocess(data_dir):
    users, dataset = {}, {}
    user_idx = 0
    with bz2.BZ2File(data_dir+'/RC_2017-11.bz2', 'r') as f:
        for line in tqdm(f):
            line = json.loads(line.rstrip())
            user = line['author']
            if user not in users.keys():
                users[user] = user_idx
                dataset[user_idx] = {
                    'num_data': 1,
                    'user_id': user,
                    'subreddit': [line['subreddit']],
                    'text': [line['body']],
                    'label': [int(line['controversiality'])]
                }
                user_idx += 1
            else:
                dataset[users[user]]['num_data'] += 1
                dataset[users[user]]['subreddit'].append(line['subreddit'])
                dataset[users[user]]['text'].append(line['body'])
                dataset[users[user]]['label'].append(line['controversiality'])

    print(len(users.keys()), len(dataset.keys()))

    num_data_per_clients = [dataset[x]['num_data'] for x in dataset.keys()]
    print(min(num_data_per_clients), max(num_data_per_clients), np.mean(num_data_per_clients),
          np.median(num_data_per_clients))

    np.random.seed(0)
    select_users_indices = np.random.randint(len(users.keys()), size=8000).tolist()

    final_dataset = {}
    new_idx = 0
    for user_id, user_idx in tqdm(users.items()):
        # dataset[user_idx]['num_data'] = len(dataset[user_idx]['label'])
        # preprocess 1
        if user_idx in select_users_indices:
            _data = dataset[user_idx]
            # print(user_idx, _data['num_data'], [x[:5] for x in _data['text']], user_id, _data['subreddit'])
            # preprocess 2
            if _data['num_data'] <= 100:
                select_idx = []
                for idx in range(_data['num_data']):
                    # preprocess 3-4
                    if user_id != _data['subreddit'][idx] and _data['text'] != '':
                        select_idx.append(idx)
                if len(select_idx) > 0:
                    final_dataset[new_idx] = {
                        'user_id': user_id,
                        'num_data': len(select_idx),
                        'text': np.array(_data['text'])[select_idx].tolist(),
                        'label': np.array(_data['label'])[select_idx].tolist()
                    }
                    new_idx += 1

    num_clients = len(final_dataset.keys())
    print(num_clients)

    train_dataset, test_dataset = {}, {}

    for client_idx in tqdm(range(num_clients), desc='>> Split data to clients'):
        local_data = final_dataset[client_idx]
        user_train_data_num = local_data['num_data']

        # split train, test in local data
        num_train = int(0.9 * user_train_data_num) if user_train_data_num >= 10 else user_train_data_num
        num_test = user_train_data_num - num_train if user_train_data_num >= 10 else 0

        if user_train_data_num >= 10:
            np.random.seed(client_idx)
            train_indices = np.random.choice(user_train_data_num, num_train, replace=False).tolist()
            test_indices = list(set(np.arange(user_train_data_num)) - set(train_indices))

            train_dataset[client_idx] = {'datasize': num_train,
                                         'text': np.array(local_data['text'])[train_indices].tolist(),
                                         'label': np.array(local_data['label'])[train_indices].tolist()}
            test_dataset[client_idx] = {'datasize': num_test,
                                        'text': np.array(local_data['text'])[test_indices].tolist(),
                                        'label': np.array(local_data['label'])[test_indices].tolist()}
        else:
            train_dataset[client_idx] = {'datasize': num_train, 'text': local_data['text'],
                                         'label': local_data['label']}

    train_data_num, test_data_num = 0, 0
    train_data_local_dict, test_data_local_dict = {}, {}
    train_data_local_num_dict, test_data_local_num_dict = {}, {}
    test_clients = test_dataset.keys()

    for client_idx in tqdm(range(len(train_dataset.keys())), desc='>> Split data to clients'):
        train_data = train_dataset[client_idx]
        training_data = _batch_data(train_data)

        train_data_local_dict[client_idx] = training_data
        train_data_local_num_dict[client_idx] = train_data['datasize']

        if client_idx in test_clients:
            test_data = test_dataset[client_idx]
            testing_data = _batch_data(test_data)
            test_data_local_dict[client_idx] = testing_data
            test_data_local_num_dict[client_idx] = test_data['datasize']

    final_final_dataset = {}
    final_final_dataset['train'] = {
        'data_sizes': train_data_local_num_dict,
        'data': train_data_local_dict,
    }
    final_final_dataset['test'] = {
        'data_sizes': test_data_local_num_dict,
        'data': test_data_local_dict,
    }

    with open(data_dir+'/Reddit_preprocessed_7668.pickle', 'wb') as f:
        pickle.dump(final_final_dataset, f)

    return final_final_dataset



def _batch_data(data, batch_size=128, maxlen=400):
    '''
    data is a dict := {'x': [numpy array], 'y': [numpy array]} (on one client)
    returns x, y, which are both numpy array of length: batch_size
    '''
    data_x = np.array(data['text'])
    data_y = np.array(data['label'])

    # randomly shuffle data
    np.random.seed(0)
    rng_state = np.random.get_state()
    np.random.shuffle(data_x)
    np.random.set_state(rng_state)
    np.random.shuffle(data_y)

    # loop through mini-batches
    batch_data = list()
    for i in range(0, len(data_x), batch_size):
        batched_x = data_x[i:i + batch_size]
        batched_y = data_y[i:i + batch_size]
        batched_x = _process_x(batched_x, maxlen)
        batched_y = torch.tensor(batched_y, dtype=torch.long)
        batch_data.append((batched_x, batched_y))
    return batch_data#, maxlen_lst


def _process_x(raw_x_batch, maxlen=400):
    CHAR_VOCAB = list('dhlptx@DHLPTX $(,048cgkoswCGKOSW[_#\'/37;?bfjnrvzBFJNRVZ"&*.26:\naeimquyAEIMQUY]!%)-159\r')
    ALL_LETTERS = "".join(CHAR_VOCAB)

    x_batch = []
    for word in raw_x_batch:
        indices = torch.empty((0,), dtype=torch.long)
        for c in word:
            tmp = ALL_LETTERS.find(c)
            tmp = len(ALL_LETTERS) if tmp == -1 else tmp
            tmp = torch.tensor([tmp], dtype=torch.long)
            indices = torch.cat((indices, tmp), dim=0)
        x_batch.append(indices)

    x_batch2 = torch.empty((0, maxlen), dtype=torch.long)
    for x in x_batch:
        x = torch.unsqueeze(F.pad(x, (0, maxlen-x.size(0)), value=maxlen-1), 0)
        x_batch2 = torch.cat((x_batch2, x), dim=0)
    return x_batch2
