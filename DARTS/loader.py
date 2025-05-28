from __future__ import print_function
import os
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from utils import padding_zeros_darts, get_bit_data_darts
import pickle
import nasbench.api as api
import numpy as np
import random
from darts import DataSetDarts
from utils import get_matrix_data_darts
import darts
from genotypes import Genotype, PRIMITIVES

from collections import namedtuple
Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')
def convert_to_genotype(arch):
  op_dict = {
    0: 'none',
    1: 'sep_conv_5x5',
    2: 'dil_conv_5x5',
    3: 'sep_conv_3x3',
    4: 'dil_conv_3x3',
    5: 'max_pool_3x3',
    6: 'avg_pool_3x3',
    7: 'skip_connect'
  }
  darts_arch = [[], []]
  i = 0
  for cell in arch:
    for n in cell:
      darts_arch[i].append((op_dict[n[1]], n[0]))
    i += 1
  geno = Genotype(normal=darts_arch[0], normal_concat=[2, 3, 4, 5], reduce=darts_arch[1], reduce_concat=[2, 3, 4, 5])
  return geno

# 映射字典
op_list = {
    'input1': 0,
    'input2': 1,
    'output': 2,
    'sep_conv_3x3': 3,
    'sep_conv_5x5': 4,
    'dil_conv_3x3': 5,
    'dil_conv_5x5': 6,
    'avg_pool_3x3': 7,
    'max_pool_3x3': 8,
    'skip_connect': 9,
    'none': 10
}

def darts_to_nasbench101(genotype):
  arch = []
  for arch_list, concat in [(genotype.normal, genotype.normal_concat), (genotype.reduce, genotype.reduce_concat)]:
    num_ops = len(arch_list) + 3
    adj = np.zeros((num_ops, num_ops), dtype=np.uint8)
    ops = ['input1', 'input2', 'output']
    node_lists = [[0], [1], [2, 3], [4, 5], [6, 7], [8, 9], [10]]
    for node in arch_list:
      node_idx = len(ops) - 1
      adj[node_lists[node[1]], node_idx] = 1
      ops.insert(-1, node[0])
    adj[[x for c in concat for x in node_lists[c]], -1] = 1
    cell = {'adj': adj,
            'ops': ops,
            }
    arch.append(cell)
  adj = np.zeros((num_ops * 2, num_ops * 2), dtype=np.uint8)
  adj[:num_ops, :num_ops] = arch[0]['adj']
  adj[num_ops:, num_ops:] = arch[1]['adj']
  ops = arch[0]['ops'] + arch[1]['ops']
  # 使用列表推导式将 input_list 中的每个元素替换成对应的数字
  ops_int = [op_list[element] for element in ops]
  arch = {'adj': adj,
          'ops': ops_int}
  return arch

def sample_darts_arch(available_ops):
  geno = []
  for _ in range(2):
    cell = []
    for i in range(4):
      ops_normal = np.random.choice(available_ops, 2)
      nodes_in_normal = sorted(np.random.choice(range(i + 2), 2, replace=False))
      cell.extend([(ops_normal[0], nodes_in_normal[0]), (ops_normal[1], nodes_in_normal[1])])
    geno.append(cell)
  genotype = Genotype(normal=geno[0], normal_concat=[2, 3, 4, 5], reduce=geno[1], reduce_concat=[2, 3, 4, 5])
  return genotype

# 定义op_dict和convert_to_arch函数
def convert_to_arch(geno):
    op_dict = {
        'none': 0,
        'sep_conv_5x5': 1,
        'dil_conv_5x5': 2,
        'sep_conv_3x3': 3,
        'dil_conv_3x3': 4,
        'max_pool_3x3': 5,
        'avg_pool_3x3': 6,
        'skip_connect': 7
    }

    # Initialize arch
    normal_cell = []
    reduce_cell = []

    # For the normal and reduce paths
    for path, cell in zip([geno.normal, geno.reduce], [normal_cell, reduce_cell]):
        for op, pos in path:
            op_index = op_dict[op]  # Map operation name to index
            cell.append((pos, op_index))  # (position, operation_index)

    # Return a tuple of normal and reduce
    return (normal_cell, reduce_cell)

def get_meta_train_loader(batch_size, data_path, data_name, num_class='None'):
  dataset = MetaTrainDataset()
  print(f'==> Meta-Train dataset {data_name}')
  print(f'==> The number of tasks for meta-training: {len(dataset)}')

  loader = DataLoader(dataset=dataset,
                      batch_size=batch_size,
                      shuffle=True,
                      num_workers=0,#1
                      collate_fn=collate_fn)
  return loader


def get_meta_test_loader(batch_size, data_path, data_name, num_class=None):
  dataset = MetaTestDataset()
  print(f'==> Meta-Test dataset {data_name}')
  print(f'==> The number of tasks for meta-testing: {len(dataset)}')

  loader = DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0,#1
                          collate_fn=collate_fn1)
  return loader



# 加载元测试数据集并返回样本
class MetaTrainDataset(Dataset):
  def __init__(self, mode='train'):
    self.dataset = 'darts'
    self.acc_norm = True

    save_path = os.path.join('path', 'darts_data.pkl')
    print("Downloading darts data...")
    self.darts_data = torch.load(save_path)
    self.dataset = self.darts_data['dataset']
    self.acc = self.darts_data['best_acc_list']
    self.acc = torch.tensor(self.acc, dtype=torch.float32)

    self.mean = torch.mean(self.acc).item()
    self.std = torch.std(self.acc).item()

    self.arch = []
    for dataset in self.dataset:
        genotype = convert_to_genotype(dataset)
        arch = darts_to_nasbench101(genotype)
        self.arch.append(arch)

    self.mode = mode
    self.idx_lst = self._split_data()

  def _split_data(self):
    random_idx_lst = list(range(len(self.acc)))
    random.shuffle(random_idx_lst)

    idx_dict = {}
    total_size = len(self.acc)
    valid_size = total_size
    idx_dict['valid'] = random_idx_lst[valid_size-20:]
    idx_dict['train'] = random_idx_lst[:valid_size]
    return idx_dict

  def __len__(self):
    return len(self.idx_lst[self.mode])

  def set_mode(self, mode):
    self.mode = mode

  def __getitem__(self, index):
    idx = self.idx_lst[self.mode][index]
    data = self.arch[idx]
    acc = self.acc[idx]
    if self.acc_norm:
      acc = ((acc - self.mean) / self.std) / 100.0
    else:
      acc = acc / 100.0

    return data, acc

class MetaTestDataset(Dataset):
  def __init__(self, sample_range=100000):
    self.sample_range = sample_range
    self.dataset = 'darts'

    if os.path.exists(os.path.join('path', 'darts_test_data3.pt')):
      self.dataset = torch.load(os.path.join('path', 'darts_test_data3.pt'))
      self.arch = []
      for dataset in self.dataset:
        genotype = convert_to_genotype(dataset)
        arch = darts_to_nasbench101(genotype)
        self.arch.append(arch)
    else:
      self.Darts = DataSetDarts(100000)
      self.dataset = self.Darts.dataset
      self.arch = []
      for dataset in self.dataset:
        genotype = convert_to_genotype(dataset)
        arch = darts_to_nasbench101(genotype)
        self.arch.append(arch)

  def __len__(self):
    return len(self.arch)

  def __getitem__(self, index):
    data = self.arch[index]
    darts = self.dataset[index]
    return data, darts

def collate_fn(batch):
  graph = [item[0] for item in batch]
  acc = torch.stack([item[1] for item in batch])
  return [graph, acc]

def collate_fn1(batch):
  graph = [item[0] for item in batch]
  darts = [item[1] for item in batch]
  return [graph,darts]

