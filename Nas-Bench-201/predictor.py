from __future__ import print_function
import torch
import os
import random
from tqdm import tqdm
import numpy as np
import time
import os
import math

from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from scipy.stats import pearsonr, kendalltau

from utils import Log, get_log
from utils import load_model, save_model
from loader import get_meta_train_loader, get_meta_test_loader
from predictor_model import PredictorModel


class Predictor:
  def __init__(self, args):
    self.args = args
    self.batch_size = args.batch_size
    self.data_path = args.data_path
    self.num_sample = args.num_sample
    self.max_epoch = args.max_epoch
    self.save_epoch = args.save_epoch
    self.model_path = args.model_path
    self.save_path = args.save_path
    self.model_name = args.model_name
    self.data_name = args.data_name
    self.num_class = args.num_class
    self.load_epoch = args.load_epoch
    self.test = args.test
    self.device = torch.device("cuda:0")
    self.max_corr_dict = {'corr': -1, 'epoch': -1, 'loss':-1, 'tau':-1}
    self.train_arch = args.train_arch

    self.model = PredictorModel(args, self.device)
    self.model.to(self.device)

    if self.test:
      load_model(self.model, self.model_path, load_max_pt='ckpt_max_corr.pt')# ckpt_max_corr.pt
      self.mtrloader = get_meta_test_loader(self.batch_size, self.data_path, self.data_name)
      self.acc_mean = self.mtrloader.dataset.mean
      self.acc_std = self.mtrloader.dataset.std

      self.mtrlog = Log(self.args, open(os.path.join(self.save_path, self.model_name, 'meta_test_predictor.log'), 'w'))
      self.mtrlog.print_args()

    else:
      self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4, weight_decay=1e-4)
      self.scheduler = ReduceLROnPlateau(self.optimizer, 'max', factor=0.1, patience=10, verbose=True)
      self.mtrloader =  get_meta_train_loader(self.batch_size, self.data_path, self.data_name)

      self.acc_mean = self.mtrloader.dataset.mean
      self.acc_std = self.mtrloader.dataset.std

      self.mtrlog = Log(self.args, open(os.path.join(self.save_path, self.model_name, 'meta_train_predictor.log'), 'w'))
      self.mtrlog.print_args()

  def forward(self, arch):
    numv, adj, out = self.model.get_arch(arch)
    G_mu = self.model.get_arch_features(numv, adj, out)
    y_pred = self.model.predictor(G_mu)
    return y_pred , adj, out

  def meta_train(self):
    for epoch in range(1, self.max_epoch + 1):
      self.mtrlog.ep_sttime = time.time()

      ######## train #######
      loss, corr, tau = self.meta_train_epoch(epoch)
      if math.isnan(loss) or math.isinf(loss):
        print(f"NaN or Inf in loss: {loss}")
      self.scheduler.step(tau)
      self.mtrlog.print_pred_log(loss, corr, tau, 'train', epoch)

      ######## valid #######
      valoss, vacorr, vatau = self.meta_validation(epoch)
      if self.max_corr_dict['tau'] < vatau:
        self.max_corr_dict['tau'] = vatau
        self.max_corr_dict['corr'] = vacorr
        self.max_corr_dict['epoch'] = epoch
        self.max_corr_dict['loss'] = valoss
        save_model(epoch, self.model, self.model_path, max_corr=True)
      self.mtrlog.print_pred_log(valoss, vacorr, vatau, 'valid', max_corr_dict=self.max_corr_dict)

      if epoch % self.save_epoch == 0:
        save_model(epoch, self.model, self.model_path)

    self.mtrlog.save_time_log()
    self.mtrlog.max_corr_log(self.max_corr_dict)

  def meta_train_epoch(self, epoch):
    self.model.to(self.device)
    self.model.train()
    self.mtrloader.dataset.set_mode('train')

    dlen = len(self.mtrloader.dataset)
    trloss = 0
    y_all, y_pred_all =[], []
    pbar = tqdm(self.mtrloader, ncols=200)

    for g, acc in pbar:
      self.optimizer.zero_grad()
      y_pred, adj, out = self.forward(g)
      y_pred = y_pred.to(self.device)
      y = acc.to(self.device)
      loss = self.model.loss(y_pred, y, adj, out)  #If BPRLoss:  loss = self.model.loss(y_pred, y)
      loss.backward()
      self.optimizer.step()

      y = y.tolist()
      y_pred = y_pred.squeeze().tolist()
      y_all += y
      y_pred_all += y_pred
      pbar.set_description(get_log(epoch, loss, y_pred, y, self.acc_std['train'], self.acc_mean['train']))
      trloss += float(loss)

    return trloss/dlen, pearsonr(np.array(y_all), np.array(y_pred_all))[0], kendalltau(np.array(y_all*10), np.array(y_pred_all*10))[0]

  def meta_validation(self, epoch):
    self.model.to(self.device)
    self.model.eval()

    valoss = 0
    self.mtrloader.dataset.set_mode('valid')
    dlen = len(self.mtrloader.dataset)
    y_all, y_pred_all =[], []
    pbar = tqdm(self.mtrloader, ncols=200)

    with torch.no_grad():
      for g, acc in pbar:
        y_pred, adj, out = self.forward(g)
        y = acc.to(self.device)
        loss = self.model.loss(y_pred, y, adj, out) #If BPRLoss:  loss = self.model.loss(y_pred, y)
        y = y.tolist()
        y_pred = y_pred.squeeze().tolist()
        y_all += y
        y_pred_all += y_pred
        pbar.set_description(get_log(epoch, loss, y_pred, y, self.acc_std['valid'], self.acc_mean['valid'], tag='val'))
        valoss += float(loss)

    return valoss/dlen, pearsonr(np.array(y_all), np.array(y_pred_all))[0], kendalltau(np.array(y_all*10), np.array(y_pred_all*10))[0]

  def meta_test1(self):
    for epoch in range(1, 2):
      self.mtrlog.ep_sttime = time.time()
      ######## test #######
      valoss, vacorr, vatau = self.meta_test_epoch(epoch)
      if self.max_corr_dict['tau'] < vatau:
        self.max_corr_dict['tau'] = vatau
        self.max_corr_dict['corr'] = vacorr
        self.max_corr_dict['epoch'] = epoch
        self.max_corr_dict['loss'] = valoss
      self.mtrlog.print_pred_log(valoss, vacorr, vatau, 'test', max_corr_dict=self.max_corr_dict)

    self.mtrlog.save_time_log()
    self.mtrlog.max_corr_log(self.max_corr_dict)

  def meta_test_epoch(self, epoch):
    self.model.to(self.device)
    self.model.eval()

    valoss = 0
    dlen = len(self.mtrloader.dataset)
    y_all, y_pred_all = [], []
    pbar = tqdm(self.mtrloader, ncols=200)

    with torch.no_grad():
      for g, acc in pbar:
        y_pred, adj, out = self.forward(g)
        y = acc.to(self.device)
        loss = self.model.loss(y_pred, y, adj, out)
        y = y.tolist()
        y_pred = y_pred.squeeze().tolist()
        y_all += y
        if type(y_pred)==float:
          y_pred = [y_pred]
        y_pred_all += y_pred
        pbar.set_description(get_log(epoch, loss, y_pred, y, self.acc_std, self.acc_mean, tag='val'))
        valoss += float(loss)

    avg_valoss = valoss / dlen
    pearson_corr = pearsonr(np.array(y_all), np.array(y_pred_all))[0]
    kendall_corr = kendalltau(np.array(y_all * 10), np.array(y_pred_all * 10))[0]
    pred_acc_pairs = list(zip(y_pred_all, y_all))
    top_50_pred_archs = sorted(pred_acc_pairs, key=lambda x: x[0], reverse=True)[:50]
    top_50_pred_archs_sorted_by_true_acc = sorted(top_50_pred_archs, key=lambda x: x[1], reverse=True)

    for idx, (pred, true) in enumerate(top_50_pred_archs_sorted_by_true_acc, 1):
      true_acc = true * 100 * self.acc_std + self.acc_mean
      print(f"Rank {idx}: True Acc: {true_acc:.4f}")

    return avg_valoss, pearson_corr, kendall_corr









