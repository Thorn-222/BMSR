from __future__ import print_function
import os
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pickle
import nasbench.api as api
import numpy as np
import random

def get_meta_train_loader(batch_size, data_path, data_name, num_class='None',RE=False,history=None):
  dataset = MetaTrainDataset(RE=RE,history=history)
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
                          collate_fn=collate_fn)
  return loader


NASBENCH_TFRECORD = os.path.join('path', 'nasbench_only108.tfrecord')

INPUT = 'input'
OUTPUT = 'output'
CONV1X1 = 'conv1x1-bn-relu'
CONV3X3 = 'conv3x3-bn-relu'
MAXPOOL3X3 = 'maxpool3x3'
NULL = 'null'

class MetaTrainDataset(Dataset):
  def __init__(self,mode='train',RE=False,history=None):
    self.dataset = 'nas-bench-101'
    self.acc_norm = True

    # RE Training
    if(RE and history!=None):
      self.important_metrics = {}
      for iter_num, cur in enumerate(history):
        self.important_metrics[iter_num] = cur[0]
      self.acc = []
      for iter_num in self.important_metrics:
        final_valid_accuracy = self.important_metrics[iter_num]['final_test_accuracy']
        self.acc.append(final_valid_accuracy)
      self.acc = torch.tensor(self.acc, dtype=torch.float32)
    else:
      # use the len of index_list and the first index to distinguish different index_list
      save_path = os.path.join('path', 'tiny_nas_bench_101_test.pkl')
      if not os.path.isfile(save_path):
        nasbench = api.NASBench(NASBENCH_TFRECORD)
        self.important_metrics = {}
        for iter_num, unique_hash in enumerate(nasbench.hash_iterator()):
          fixed_metrics, computed_metrics = nasbench.get_metrics_from_hash(unique_hash)
          final_training_time_list = []
          final_valid_accuracy_list = []
          final_test_accuracy_list = []
          for i in range(3):
            # the 108 means the train epochs, and I have only downloaded the metrics in Epoch 108 (full train)
            # the three iterations: three results of independent experiments recorded in the dataset
            final_training_time_list.append(computed_metrics[108][i]['final_training_time'])
            final_valid_accuracy_list.append(computed_metrics[108][i]['final_validation_accuracy'])
            final_test_accuracy_list.append(computed_metrics[108][i]['final_test_accuracy'])
          # use the mean of three metrics
          final_training_time = np.mean(final_training_time_list)
          final_valid_accuracy = np.mean(final_valid_accuracy_list)
          final_test_accuracy = np.mean(final_test_accuracy_list)

          # using the index to create dicts
          self.important_metrics[iter_num] = {}
          self.important_metrics[iter_num]['fixed_metrics'] = fixed_metrics
          self.important_metrics[iter_num]['final_training_time'] = final_training_time
          self.important_metrics[iter_num]['final_valid_accuracy'] = final_valid_accuracy
          self.important_metrics[iter_num]['final_test_accuracy'] = final_test_accuracy

        if not os.path.isdir('pkl'):
          os.mkdir('pkl')

        with open(save_path, 'wb') as file:
          pickle.dump(self.important_metrics, file)
      else:
        with open(save_path, 'rb') as file:
          self.important_metrics = pickle.load(file)

      self.important_metrics = self.operations2integers(self.important_metrics)

      # acc file
      acc_file_path = 'acc_nas_bench_101.pt'
      if os.path.exists(acc_file_path):
        self.acc = torch.load(acc_file_path)
      else:
        self.acc = []
        for iter_num in self.important_metrics:
          final_valid_accuracy = self.important_metrics[iter_num]['final_valid_accuracy']
          self.acc.append(final_valid_accuracy)

        self.acc = torch.tensor(self.acc, dtype=torch.float32)
        torch.save(self.acc, acc_file_path)

    self.mean = torch.mean(self.acc).item()
    self.std = torch.std(self.acc).item()

    # Partitioning Datasets Based on Schema
    self.mode = mode
    self.idx_lst = self._split_data()

  def operations2integers(self, important_metrics):
    dict_oper2int = {NULL: 0, CONV1X1: 1, CONV3X3: 2, MAXPOOL3X3: 3}
    for i in important_metrics:
      fix_metrics = important_metrics[i]['fixed_metrics']
      module_operations = fix_metrics['module_operations']
      module_integers = np.array([dict_oper2int[x] for x in module_operations[1: -1]])
      # Add 0 at the first position, indicating that the operation from input->node 0 is none
      module_integers = np.insert(module_integers, 0, 0)
      module_integers = np.append(module_integers, 0)
      important_metrics[i]['fixed_metrics']['module_integers'] = module_integers
    return important_metrics

  def _split_data(self):
    random_idx_lst = list(range(len(self.important_metrics)))
    random.shuffle(random_idx_lst)

    idx_dict = {}
    total_size = len(self.important_metrics)
    valid_size = total_size//1000+1
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
    metrics = self.important_metrics[idx]
    acc = self.acc[idx]
    if self.acc_norm:
      acc = ((acc - self.mean) / self.std) / 100.0
    else:
      acc = acc / 100.0

    return metrics, acc

class MetaTestDataset(Dataset):
  def __init__(self):
    self.dataset = 'nas-bench-101'
    self.acc_norm = True

    # use the len of index_list and the first index to distinguish different index_list
    save_path = os.path.join('path', 'tiny_nas_bench_101.pkl')
    if not os.path.isfile(save_path):
      nasbench = api.NASBench(NASBENCH_TFRECORD)
      self.important_metrics = {}
      for iter_num, unique_hash in enumerate(nasbench.hash_iterator()):
        fixed_metrics, computed_metrics = nasbench.get_metrics_from_hash(unique_hash)
        final_training_time_list = []
        final_valid_accuracy_list = []
        final_test_accuracy_list = []
        for i in range(3):
          # the 108 means the train epochs, and I have only downloaded the metrics in Epoch 108 (full train)
          # the three iterations: three results of independent experiments recorded in the dataset
          final_training_time_list.append(computed_metrics[108][i]['final_training_time'])
          final_valid_accuracy_list.append(computed_metrics[108][i]['final_validation_accuracy'])
          final_test_accuracy_list.append(computed_metrics[108][i]['final_test_accuracy'])
        # use the mean of three metrics
        final_training_time = np.mean(final_training_time_list)
        final_valid_accuracy = np.mean(final_valid_accuracy_list)
        final_test_accuracy = np.mean(final_test_accuracy_list)

        # using the index to create dicts
        self.important_metrics[iter_num] = {}
        self.important_metrics[iter_num]['fixed_metrics'] = fixed_metrics
        self.important_metrics[iter_num]['final_training_time'] = final_training_time
        self.important_metrics[iter_num]['final_valid_accuracy'] = final_valid_accuracy
        self.important_metrics[iter_num]['final_test_accuracy'] = final_test_accuracy

      if not os.path.isdir('pkl'):
        os.mkdir('pkl')

      with open(save_path, 'wb') as file:
        pickle.dump(self.important_metrics, file)
    else:
      with open(save_path, 'rb') as file:
        self.important_metrics = pickle.load(file)

    self.important_metrics = self.operations2integers(self.important_metrics)

    acc_file_path = 'test_acc_nas_bench_101.pt'
    if os.path.exists(acc_file_path):
        self.acc = torch.load(acc_file_path)
    else:
        self.acc = []
        for iter_num in self.important_metrics:
            final_valid_accuracy = self.important_metrics[iter_num]['final_test_accuracy']
            self.acc.append(final_valid_accuracy)

        self.acc = torch.tensor(self.acc, dtype=torch.float32)
        torch.save(self.acc, acc_file_path)

    self.mean = torch.mean(self.acc).item()
    self.std = torch.std(self.acc).item()

  def operations2integers(self, important_metrics):
    dict_oper2int = {NULL: 0, CONV1X1: 1, CONV3X3: 2, MAXPOOL3X3: 3}
    for i in important_metrics:
      fix_metrics = important_metrics[i]['fixed_metrics']
      module_operations = fix_metrics['module_operations']
      module_integers = np.array([dict_oper2int[x] for x in module_operations[1: -1]])
      module_integers = np.insert(module_integers, 0, 0)
      module_integers = np.append(module_integers, 0)
      important_metrics[i]['fixed_metrics']['module_integers'] = module_integers
    return important_metrics

  def __len__(self):
    return len(self.important_metrics)

  def __getitem__(self, index):
    metrics = self.important_metrics[index]
    acc = self.acc[index]
    if self.acc_norm:
      acc = ((acc - self.mean) / self.std) / 100.0
    else:
      acc = acc / 100.0

    return metrics, acc

def collate_fn(batch):
  graph = [item[0] for item in batch]
  acc = torch.stack([item[1] for item in batch])
  return [graph, acc]





