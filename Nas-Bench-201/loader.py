from __future__ import print_function
import os
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import random

def get_meta_train_loader(batch_size, data_path, data_name, acc_type='valid'):
  dataset = MetaTrainDataset(data_path, data_name, acc_type=acc_type)
  print(f'==> Meta-Train dataset {data_name}')
  print(f'==> The number of tasks for meta-training: {len(dataset)}')

  loader = DataLoader(dataset=dataset,
                      batch_size=batch_size,
                      shuffle=True,
                      num_workers=0,
                      collate_fn=collate_fn)
  return loader


def get_meta_test_loader(batch_size, data_path, data_name, acc_type='test'):
  dataset = MetaTestDataset(data_path, data_name,acc_type=acc_type)
  print(f'==> Meta-Test dataset {data_name}')
  print(f'==> The number of tasks for meta-testing: {len(dataset)}')

  loader = DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0,
                          collate_fn=collate_fn)
  return loader

class MetaTestDataset_yuan(Dataset):
  def __init__(self, data_path, data_name, num_class=None):
    self.data_name = data_name
    self.acc_norm = True

    num_class_dict = {
      'cifar100': 100,
      'cifar10': 10,
      'imagenet16-120':120,
      'mnist': 10,
      'svhn': 10,
      'aircraft': 30,
      'pets': 37
    }
    if num_class is not None:
      self.num_class = num_class
    else:
      self.num_class = num_class_dict[data_name]

    self.g = torch.load(os.path.join(data_path, 'nasbench201.pt'))['arch']['igraph']
    self.acc = torch.load(os.path.join(data_path, 'nasbench201.pt'))['test-acc'][data_name]
    self.acc = torch.tensor(self.acc, dtype=torch.float32)
    self.mean = torch.mean(self.acc).item()
    self.std = torch.std(self.acc).item()


  def __len__(self):
    return len(self.g)

  def __getitem__(self, index):
    graph = self.g[index]
    acc = self.acc[index]

    if self.acc_norm:
      acc = ((acc - self.mean) / self.std) / 100.0
    else:
      acc = acc / 100.0

    return graph, acc

class MetaTrainDataset_yuan(Dataset):
  def __init__(self, data_path, data_name, num_class=None, mode='train'):
    self.data_name = data_name
    self.acc_norm = True

    num_class_dict = {
      'cifar100': 100,
      'cifar10': 10,
      'mnist': 10,
      'svhn': 10,
      'aircraft': 30,
      'pets': 37
    }
    if num_class is not None:
      self.num_class = num_class
    else:
      self.num_class = num_class_dict[data_name]

    self.g = torch.load(os.path.join(data_path, 'nasbench201.pt'))['arch']['igraph']
    self.acc = torch.load(os.path.join(data_path, 'nasbench201.pt'))['test-acc'][data_name]
    self.acc = torch.tensor(self.acc, dtype=torch.float32)
    self.mean = torch.mean(self.acc).item()
    self.std = torch.std(self.acc).item()

    self.mode = mode
    self.idx_lst = self._split_data()

  def _split_data(self):
    random_idx_lst = list(range(len(self.g)))
    random.shuffle(random_idx_lst)

    idx_dict = {}
    total_size = len(self.g)
    valid_size = total_size //10+1
    valid_size_1 = total_size -400
    idx_dict['valid'] = random_idx_lst[valid_size_1:valid_size_1+200]
    idx_dict['train'] = random_idx_lst[:valid_size]
    return idx_dict

  def __len__(self):
    return len(self.idx_lst[self.mode])

  def set_mode(self, mode):
    self.mode = mode

  def __getitem__(self, index):
    idx = self.idx_lst[self.mode][index]
    graph = self.g[idx]
    acc = self.acc[idx]
    if self.acc_norm:
      acc = ((acc - self.mean) / self.std) / 100.0
    else:
      acc = acc / 100.0

    return graph, acc

class MetaTrainDataset(Dataset):
  def __init__(self, data_path, data_name, mode='train', acc_type='valid'):
    self.data_name = data_name
    self.acc_norm = True

    self.nasbench201_dict = np.load(os.path.join(data_path, 'nasbench201_dict_search.npy'), allow_pickle=True).item()
    if acc_type == 'test':
      self.acc = [v[f"{self.data_name}_test"] for v in self.nasbench201_dict.values()]
    elif acc_type == 'valid':
      self.acc = [v[f"{self.data_name}_valid"] for v in self.nasbench201_dict.values()]
    self.acc = torch.tensor(self.acc, dtype=torch.float32)

    self.mode = mode
    self.idx_lst = self._split_data()

    self.mean = {}
    self.std = {}

    self.mean['train'] = torch.mean(self.acc[self.idx_lst['train']]).item()
    self.std['train'] = torch.std(self.acc[self.idx_lst['train']]).item()
    self.mean['valid'] = torch.mean(self.acc[self.idx_lst['valid']]).item()
    self.std['valid'] = torch.std(self.acc[self.idx_lst['valid']]).item()

  def _split_data(self):
    random_idx_lst = list(range(len(self.nasbench201_dict)))
    random.shuffle(random_idx_lst)

    idx_dict = {}
    total_size = len(self.nasbench201_dict)
    valid_size = total_size //100
    #valid_size = total_size - 15575
    valid_size_1 = total_size - 8000
    idx_dict['valid'] = random_idx_lst[valid_size_1:valid_size_1+200]
    idx_dict['train'] = random_idx_lst[:valid_size]
    return idx_dict

  def __len__(self):
    return len(self.idx_lst[self.mode])

  def set_mode(self, mode):
    self.mode = mode

  def parse_arch_str(self,arch_str):
    arch_list = []
    node_strs = arch_str.strip('|').split('+')
    for node_str in node_strs:
      node = []
      items = node_str.strip('|').split('|')
      for item in items:
        if '~' in item:
          op, idx = item.split('~')
          node.append((op, int(idx)))
      arch_list.append(node)
    return arch_list

  def nasbench201_to_nasbench101(self,arch_list):
      num_ops = sum(range(1, 1 + len(arch_list))) + 2
      adj = np.zeros((num_ops, num_ops), dtype=np.uint8)
      ops = ['input', 'output']
      node_lists = [[0]]
      for node_201 in arch_list:
          node_list = []
          for node in node_201:
              node_idx = len(ops) - 1
              adj[node_lists[node[1]], node_idx] = 1
              ops.insert(-1, node[0])
              node_list.append(node_idx)
          node_lists.append(node_list)
      adj[-(1 + len(arch_list)):-1, -1] = 1
      arch = {'adj': adj,
              'ops': ops, }
      return arch


  def __getitem__(self, index):
    idx = self.idx_lst[self.mode][index]
    arch = self.nasbench201_dict[str(idx)]['arch']
    acc = self.acc[idx]
    if self.acc_norm:
      acc = ((acc - self.mean[self.mode]) / self.std[self.mode]) / 100.0
    else:
      acc = acc / 100.0

    return arch, acc

class MetaTestDataset(Dataset):
  def __init__(self, data_path, data_name, acc_type='test'):
    self.data_name = data_name
    self.acc_norm = True

    self.nasbench201_dict = np.load(os.path.join(data_path, 'nasbench201_dict_search.npy'), allow_pickle=True).item()
    if acc_type == 'test':
      self.acc = [v[f"{self.data_name}_test"] for v in self.nasbench201_dict.values()]
    elif acc_type == 'valid':
      self.acc = [v[f"{self.data_name}_valid"] for v in self.nasbench201_dict.values()]
    self.acc = torch.tensor(self.acc, dtype=torch.float32)
    self.mean = torch.mean(self.acc).item()
    self.std = torch.std(self.acc).item()

  def __len__(self):
    return len(self.nasbench201_dict)

  def parse_arch_str(self,arch_str):
    arch_list = []
    node_strs = arch_str.strip('|').split('+')
    for node_str in node_strs:
      node = []
      items = node_str.strip('|').split('|')
      for item in items:
        if '~' in item:
          op, idx = item.split('~')
          node.append((op, int(idx)))
      arch_list.append(node)
    return arch_list

  def nasbench201_to_nasbench101(self,arch_list):
      num_ops = sum(range(1, 1 + len(arch_list))) + 2
      adj = np.zeros((num_ops, num_ops), dtype=np.uint8)
      ops = ['input', 'output']
      node_lists = [[0]]
      for node_201 in arch_list:
          node_list = []
          for node in node_201:
              node_idx = len(ops) - 1
              adj[node_lists[node[1]], node_idx] = 1
              ops.insert(-1, node[0])
              node_list.append(node_idx)
          node_lists.append(node_list)
      adj[-(1 + len(arch_list)):-1, -1] = 1
      arch = {'adj': adj,
              'ops': ops, }
      return arch

  def __getitem__(self, index):
    arch = self.nasbench201_dict[str(index)]['arch']
    acc = self.acc[index]

    if self.acc_norm:
      acc = ((acc - self.mean) / self.std) / 100.0
    else:
      acc = acc / 100.0

    return arch, acc

def collate_fn(batch):
  graph = [item[0] for item in batch]
  acc = torch.stack([item[1] for item in batch])
  return [graph, acc]
