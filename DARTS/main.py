import os
import random
import numpy as np

import torch
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from parser import get_parser
from generator import Generator
from Single_model.darts.predictor.predictor import Predictor


def main():
  args = get_parser()
  os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
  device = torch.device("cuda:1")
  torch.cuda.manual_seed(args.seed)
  torch.manual_seed(args.seed)
  np.random.seed(args.seed)
  random.seed(args.seed)

  args.model_name = 'predictor'
  args.model_path = os.path.join(args.save_path, args.model_name, 'model')
  if not os.path.exists(args.model_path):
    os.makedirs(args.model_path)

  if args.model_name == 'generator':

    g = Generator(args)
    if args.test:
      g.meta_test()
    else:
      g.meta_train()
  elif args.model_name == 'predictor':
    p = Predictor(args)
    if args.test:
      p.meta_test1()
    else:
      p.meta_train()
  else:
    raise ValueError('You should select generator|predictor|train_arch')


if __name__ == '__main__':
  main()
