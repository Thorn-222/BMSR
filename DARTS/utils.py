from __future__ import print_function
import copy
from collections import namedtuple, OrderedDict
from torch.autograd import Variable
import torch
import torchvision.transforms as transforms
import os
import shutil
import logging
import sys
import time
import numpy as np
import scipy.stats

PADDING_MAX_LENGTH = 9

def to_categorical(labels, num_classes, dtype='int8'):
    one_hot = np.zeros((len(labels), num_classes), dtype=dtype)
    one_hot[np.arange(len(labels)), labels] = 1
    return one_hot

def delete_margin(matrix):
    return matrix[:-1, 1:]

def padding_zero_in_matrix(important_metrics, max_length=PADDING_MAX_LENGTH):
    for i in important_metrics:
        len_operations = len(important_metrics[i]['fixed_metrics']['module_operations'])
        if len_operations != max_length:
            for j in range(len_operations, max_length):
                important_metrics[i]['fixed_metrics']['module_operations'].insert(-1, 'null')

            adjecent_matrix = important_metrics[i]['fixed_metrics']['module_adjacency']
            padding_matrix = np.insert(adjecent_matrix, len_operations - 1,np.zeros([max_length - len_operations, len_operations]), axis=0)
            padding_matrix = np.insert(padding_matrix, [len_operations - 1],np.zeros([max_length, max_length - len_operations]), axis=1)
            important_metrics[i]['fixed_metrics']['module_adjacency'] = padding_matrix
    return important_metrics

def padding_zeros(matrix, op_list, max_length=PADDING_MAX_LENGTH):
    assert len(op_list) == len(matrix)
    padding_matrix = matrix
    len_operations = len(op_list)
    if not len_operations == max_length:
        for j in range(len_operations, max_length):
            op_list.insert(j - 1, 'null')
        adjecent_matrix = copy.deepcopy(matrix)
        padding_matrix = np.insert(adjecent_matrix, len_operations - 1,np.zeros([max_length - len_operations, len_operations]),axis=0)
        padding_matrix = np.insert(padding_matrix, [len_operations - 1],np.zeros([max_length, max_length - len_operations]), axis=1)

    return padding_matrix, op_list


def padding_zeros_darts(matrixes, ops, max_length=PADDING_MAX_LENGTH):
    padding_matrixes = []
    padding_ops = []
    for matrix, op in zip(matrixes, ops):
        if op is None:
            # matrix is None this case
            padding_matrix = np.zeros(shape=[max_length, max_length], dtype='int8')
            tmp_op = np.zeros(shape=max_length, dtype='int8')

            padding_matrixes.append(padding_matrix)
            padding_ops.append(tmp_op)
            continue

        len_operations = len(op)
        tmp_op = copy.deepcopy(op)
        padding_matrix = copy.deepcopy(matrix)
        if not len_operations == max_length:
            for j in range(len_operations, max_length):
                tmp_op.insert(j - 1, 0)

            padding_matrix = np.insert(padding_matrix, len_operations - 1,np.zeros([max_length - len_operations, len_operations]),axis=0)
            padding_matrix = np.insert(padding_matrix, [len_operations - 1],np.zeros([max_length, max_length - len_operations]), axis=1)
        padding_matrixes.append(padding_matrix)
        padding_ops.append(tmp_op)
    return padding_matrixes, padding_ops

def get_bit_data(important_metrics, integers2one_hot=True):
    X = []
    y = []
    for index in important_metrics:
        fixed_metrics = important_metrics[index]['fixed_metrics']
        adjacent_matrix = fixed_metrics['module_adjacency']
        module_integers = fixed_metrics['module_integers']
        accuracy = important_metrics[index]['final_valid_accuracy']

        adjacent_matrix = delete_margin(adjacent_matrix)
        flattened_adjacent = adjacent_matrix.flatten()
        input_metrics = []
        input_metrics.extend(flattened_adjacent)
        if integers2one_hot:
            module_integers = to_categorical(module_integers, 4, dtype='int8')
            module_integers = module_integers.flatten()
        input_metrics.extend(module_integers)
        X.append(input_metrics)
        y.append(accuracy)

    assert len(X) == len(y)

    return X, y

def get_bit_data_darts(important_metrics, integers2one_hot=True):
    X = []
    # each data in X consist with 4 dim rows
    for index in important_metrics:
        fixed_metrics = important_metrics[index]
        padding_norm_matrixes = fixed_metrics['padding_norm_matrixes']
        padding_norm_ops = fixed_metrics['padding_norm_ops']
        padding_reduc_matrixes = fixed_metrics['padding_reduc_matrixes']
        padding_reduc_ops = fixed_metrics['padding_reduc_ops']
        matrixes = padding_norm_matrixes + padding_reduc_matrixes
        ops = padding_norm_ops + padding_reduc_ops
        assert len(matrixes) == 4
        assert len(ops) == 4
        each_x = []  # len(each_x)==4
        for adjacency_matrix, op in zip(matrixes, ops):
            adjacency_matrix = delete_margin(adjacency_matrix)
            flattened_adjacency = adjacency_matrix.flatten()
            input_x = []
            input_x.extend(flattened_adjacency)
            if integers2one_hot:
                op = to_categorical(op, 4, dtype='int8')
                op = op.flatten()
            input_x.extend(op)
            each_x.append(input_x)
        X.append(each_x)

    return X

def get_matrix_data_darts(important_metrics):
    m, o = [], []
    for index in important_metrics:
        fixed_metrics = important_metrics[index]
        padding_norm_matrixes = fixed_metrics['padding_norm_matrixes']
        padding_norm_ops = fixed_metrics['padding_norm_ops']
        padding_reduc_matrixes = fixed_metrics['padding_reduc_matrixes']
        padding_reduc_ops = fixed_metrics['padding_reduc_ops']
        matrixes = padding_norm_matrixes + padding_reduc_matrixes
        ops = padding_norm_ops + padding_reduc_ops
        assert len(matrixes) == 4
        assert len(ops) == 4
        m.append(matrixes)
        o.append(ops)

    return m, o


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
    arch = []

    # For the normal and reduce paths
    for path in [geno.normal, geno.reduce]:
        cell = []
        for op, pos in path:
            op_index = op_dict[op]  # Map operation name to index
            cell.append((pos, op_index))  # (position, operation_index)
        arch.append(cell)

    return arch


# below is from DARTS code
def drop_path(x, drop_prob):
    if drop_prob > 0.:
        keep_prob = 1. - drop_prob
        mask = Variable(torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))
        x.div_(keep_prob)
        x.mul_(mask)
    return x


class AvgrageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        """
        Initialization of AverageMeter
        Parameters
        ----------
        name : str
            Name to display.
        fmt : str
            Format string to print the values.
        """
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = '{name}: {avg' + self.fmt + '}'
        return fmtstr.format(**self.__dict__)

class AverageMeterGroup:
    """Average meter group for multiple average meters"""

    def __init__(self):
        self.meters = OrderedDict()

    def update(self, data, n=1):
        for k, v in data.items():
            if k not in self.meters:
                self.meters[k] = AverageMeter(k, ":4f")
            self.meters[k].update(v, n=n)

    def __getattr__(self, item):
        return self.meters[item]

    def __getitem__(self, item):
        return self.meters[item]

    def __str__(self):
        return "  ".join(str(v) for v in self.meters.values())

    def summary(self):
        return "  ".join(v.summary() for v in self.meters.values())


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def _data_transforms_cifar10(args):
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    if args.cutout:
        train_transform.transforms.append(Cutout(args.cutout_length))

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    return train_transform, valid_transform


def count_parameters_in_MB(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name) / 1e6


def save_checkpoint(state, is_best, save):
    filename = os.path.join(save, 'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(save, 'model_best.pth.tar')
        shutil.copyfile(filename, best_filename)


def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.mkdir(path)
    print('Experiment dir : {}'.format(path))

    if scripts_to_save is not None:
        scripts_dir = os.path.join(path, 'scripts') 
        if not os.path.exists(scripts_dir):
            os.mkdir(scripts_dir)
        for script in scripts_to_save:
            dst_file = os.path.join(scripts_dir, os.path.basename(script))
            shutil.copyfile(script, dst_file)


def save(model, model_path):
    torch.save(model.state_dict(), model_path)


def load(model, model_path):
    model.load_state_dict(torch.load(model_path))


def get_logger():
    time_format = "%m/%d %H:%M:%S"
    fmt = "[%(asctime)s] %(levelname)s (%(name)s) %(message)s"
    formatter = logging.Formatter(fmt, time_format)
    logger = logging.getLogger()
    if logger.hasHandlers():
        logger.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger

def to_cuda(obj):
    if torch.is_tensor(obj):
        return obj.cuda()
    if isinstance(obj, tuple):
        return tuple(to_cuda(t) for t in obj)
    if isinstance(obj, list):
        return [to_cuda(t) for t in obj]
    if isinstance(obj, dict):
        return {k: to_cuda(v) for k, v in obj.items()}
    if isinstance(obj, (int, float, str)):
        return obj
    raise ValueError("'%s' has unsupported type '%s'" % (obj, type(obj)))


def padding_zero_in_matrix(important_metrics, max_length=PADDING_MAX_LENGTH):
    for i in important_metrics:
        len_operations = len(important_metrics[i]['fixed_metrics']['module_operations'])

        if len_operations != max_length:
            for j in range(len_operations, max_length):
                important_metrics[i]['fixed_metrics']['module_operations'].insert(-1, 'null')
            # print(important_metrics[i]['fixed_metrics']['module_operations'])

            adjecent_matrix = important_metrics[i]['fixed_metrics']['module_adjacency']
            
            padding_matrix = np.insert(adjecent_matrix, len_operations - 1,np.zeros([max_length - len_operations, len_operations]), axis=0)
            padding_matrix = np.insert(padding_matrix, [len_operations - 1],np.zeros([max_length, max_length - len_operations]), axis=1)
            important_metrics[i]['fixed_metrics']['module_adjacency'] = padding_matrix
    return important_metrics


def is_valid_DAG(g, START_TYPE=0, END_TYPE=1):
  res = g.is_dag()
  n_start, n_end = 0, 0
  for v in g.vs:
    if v['type'] == START_TYPE:
      n_start += 1
    elif v['type'] == END_TYPE:
      n_end += 1
    if v.indegree() == 0 and v['type'] != START_TYPE:
      return False
    if v.outdegree() == 0 and v['type'] != END_TYPE:
      return False
  return res and n_start == 1 and n_end == 1

def is_valid_NAS201(g, START_TYPE=0, END_TYPE=1):
  # first need to be a valid DAG computation graph
  res = is_valid_DAG(g, START_TYPE, END_TYPE)
  # in addition, node i must connect to node i+1
  res = res and len(g.vs['type'])==8
  res = res and not (0 in g.vs['type'][1:-1])
  res = res and not (1 in g.vs['type'][1:-1])
  return res

def decode_igraph_to_NAS201_matrix(g):
  m = [[0.0, 0.0, 0.0, 0.0],
       [0.0, 0.0, 0.0, 0.0],
       [0.0, 0.0, 0.0, 0.0],
       [0.0, 0.0, 0.0, 0.0]]
  xys = [(1, 0), (2, 0), (2, 1), (3, 0), (3, 1), (3, 2)]
  for i, xy in enumerate(xys):
    m[xy[0]][xy[1]] = float(g.vs[i + 1]['type']) - 2
  import numpy
  return numpy.array(m)


def decode_igraph_to_NAS_BENCH_201_string(gs):
  result = []

  for g in gs:
    if not is_valid_NAS201(g):
      result.append(None)
      continue

    m = decode_igraph_to_NAS201_matrix(g)

    types = ['none', 'skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3']
    result.append('|{}~0|+|{}~0|{}~1|+|{}~0|{}~1|{}~2|'.format(
      types[int(m[1][0])],
      types[int(m[2][0])], types[int(m[2][1])],
      types[int(m[3][0])], types[int(m[3][1])], types[int(m[3][2])]))

  return result


def decode_igraph_to_NAS_BENCH_201_string1(g):
  if not is_valid_NAS201(g):
    return None
  m = decode_igraph_to_NAS201_matrix(g)
  types = ['none', 'skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3']
  return '|{}~0|+|{}~0|{}~1|+|{}~0|{}~1|{}~2|'.format(
             types[int(m[1][0])],
             types[int(m[2][0])], types[int(m[2][1])],
             types[int(m[3][0])], types[int(m[3][1])], types[int(m[3][2])])


class Accumulator():
  def __init__(self, *args):
    self.args = args
    self.argdict = {}
    for i, arg in enumerate(args):
      self.argdict[arg] = i
    self.sums = [0] * len(args)
    self.cnt = 0

  def accum(self, val):
    val = [val] if type(val) is not list else val
    val = [v for v in val if v is not None]
    assert (len(val) == len(self.args))
    for i in range(len(val)):
      if torch.is_tensor(val[i]):
        val[i] = val[i].item()
      self.sums[i] += val[i]
    self.cnt += 1

  def clear(self):
    self.sums = [0] * len(self.args)
    self.cnt = 0

  def get(self, arg, avg=True):
    i = self.argdict.get(arg, -1)
    assert (i is not -1)
    if avg:
      return self.sums[i] / (self.cnt + 1e-8)
    else:
      return self.sums[i]

  def print_(self, header=None, time=None,
             logfile=None, do_not_print=[], as_int=[],
             avg=True):
    msg = '' if header is None else header + ': '
    if time is not None:
      msg += ('(%.3f secs), ' % time)

    args = [arg for arg in self.args if arg not in do_not_print]
    arg = []
    for arg in args:
      val = self.sums[self.argdict[arg]]
      if avg:
        val /= (self.cnt + 1e-8)
      if arg in as_int:
        msg += ('%s %d, ' % (arg, int(val)))
      else:
        msg += ('%s %.4f, ' % (arg, val))
    print(msg)

    if logfile is not None:
      logfile.write(msg + '\n')
      logfile.flush()

  def add_scalars(self, summary, header=None, tag_scalar=None,
                    step=None, avg=True, args=None):
    for arg in self.args:
      val = self.sums[self.argdict[arg]]
      if avg:
        val /= (self.cnt + 1e-8)
      else:
        val = val
      tag = f'{header}/{arg}' if header is not None else arg
      if tag_scalar is not None:
        summary.add_scalars(main_tag=tag,
                            tag_scalar_dict={tag_scalar: val},
                            global_step=step)
      else:
        summary.add_scalar(tag=tag,
                           scalar_value=val,
                           global_step=step)


class Log:
  def __init__(self, args, logf, summary=None):
    self.args = args
    self.logf = logf
    self.summary = summary
    self.stime = time.time()
    self.ep_sttime = None

  def print(self, logger, epoch, tag=None, avg=True):
    if tag == 'train':
      ct = time.time() - self.ep_sttime
      tt = time.time() - self.stime
      msg = f'[total {tt:6.2f}s (ep {ct:6.2f}s)] epoch {epoch:3d}'
      print(msg)
      self.logf.write(msg+'\n')
    logger.print_(header=tag, logfile=self.logf, avg=avg)

    if self.summary is not None:
      logger.add_scalars(
        self.summary, header=tag, step=epoch, avg=avg)
    logger.clear()

  def print_args(self):
    argdict = vars(self.args)
    print(argdict)
    for k, v in argdict.items():
      self.logf.write(k + ': ' + str(v) + '\n')
    self.logf.write('\n')

  def set_time(self):
    self.stime = time.time()

  def save_time_log(self):
    ct = time.time() - self.stime
    msg = f'({ct:6.2f}s) meta-training phase done'
    print(msg)
    self.logf.write(msg+'\n')

  def print_pred_log(self, loss, corr, tau, tag, epoch=None, max_corr_dict=None):
    if tag == 'train':
      ct = time.time() - self.ep_sttime
      tt = time.time() - self.stime
      msg = f'[total {tt:6.2f}s (ep {ct:6.2f}s)] epoch {epoch:3d}'
      self.logf.write(msg+'\n'); print(msg); self.logf.flush()
    #msg = f'ep {epoch:3d} ep time {time.time() - ep_sttime:8.2f} '
    #msg += f'time {time.time() - sttime:6.2f} '
    if max_corr_dict is not None:
      max_corr = max_corr_dict['corr']
      max_loss = max_corr_dict['loss']
      max_tau = max_corr_dict['tau']
      msg = f'{tag}: loss {loss:.6f} ({max_loss:.6f}) '
      msg += f'corr {corr:.4f} ({max_corr:.4f}) '
      msg += f'tau {tau:.4f} ({max_tau:.4f})'
    else:
      msg = f'{tag}: loss {loss:.6f} corr {corr:.4f} tau {tau:.4f}'
    self.logf.write(msg+'\n'); print(msg); self.logf.flush()

  def max_corr_log(self, max_corr_dict):
    corr = max_corr_dict['corr']
    loss = max_corr_dict['loss']
    tau = max_corr_dict['tau']
    epoch = max_corr_dict['epoch']
    msg = f'[epoch {epoch}] max correlation: {corr:.4f}, max tau: {tau:.4f} loss: {loss:.6f}'
    self.logf.write(msg+'\n'); print(msg); self.logf.flush()


def get_log(epoch, loss, y_pred, y, acc_std, acc_mean, tag='train', acc_norm = True):
  if(acc_norm):
    msg = f'[{tag}] Ep {epoch} loss {loss.item() / len(y):0.4f} '
    msg += f'pacc {y_pred[0]:0.4f}'
    msg += f'({y_pred[0] * 100.0 * acc_std + acc_mean:0.4f}) '
    msg += f'acc {y[0]:0.4f}({y[0] * 100 * acc_std + acc_mean:0.4f})'
  else:
    msg = f'[{tag}] Ep {epoch} loss {loss.item() / len(y):0.4f} '
    msg += f'pacc {y_pred[0]:0.4f} '
    msg += f'acc {y[0]:0.4f}'
  return msg


def load_model(model, model_path, load_epoch=None, load_max_pt=None):
  if load_max_pt is not None:
    ckpt_path = os.path.join(model_path, load_max_pt)
  else:
    ckpt_path = os.path.join(model_path, f'ckpt_{load_epoch}.pt')
  print(f"==> load model from {ckpt_path} ...")
  model.cpu()
  model.load_state_dict(torch.load(ckpt_path))


def save_model(epoch, model, model_path, max_corr=None):
  print("==> save current model...")
  if max_corr is not None:
    torch.save(model.cpu().state_dict(),
                os.path.join(model_path, 'ckpt_max_corr.pt'))
  else:
    torch.save(model.cpu().state_dict(),
                os.path.join(model_path, f'ckpt_{epoch}.pt'))



def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h