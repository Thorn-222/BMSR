import math
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

class BPRLoss(torch.nn.Module):
    def __init__(self, exp_weighted=False):
        super(BPRLoss, self).__init__()
        self.exp_weighted = exp_weighted

    def forward(self, input, target):
        N = input.size(0)
        total_loss = 0
        for i in range(N):
            indices = (target > target[i])
            x = torch.log(1 + torch.exp(-(input[indices] - input[i])))
            if self.exp_weighted:
                x = (torch.exp(target[i]) - 1) * (torch.exp(target[indices]) - 1) * x
            else:
                x = x
            total_loss += torch.sum(x)
        if self.exp_weighted:
            return 2 / (N * (math.e - 1)) ** 2 * total_loss
        else:
            return 2 / N ** 2 * total_loss

class StructAwareBPRLoss(nn.Module):
    def __init__(self, lambda_struct=0.5):
        super().__init__()
        self.lambda_struct = lambda_struct

    def forward(self, preds, targets, adj, ops):
        # 原始 BPR 排序损失
        loss = BPRLoss()
        base_loss = loss(preds, targets)

        # 结构相似性感知约束
        B = preds.size(0)
        adj_flat = adj.view(B, -1).float()

        ops_mean = ops.mean(dim=1)  # [B, D]
        ops_sim = F.cosine_similarity(ops_mean.unsqueeze(1), ops_mean.unsqueeze(0), dim=-1)

        adj_flat = adj.view(B, -1).float()
        adj_sim = F.cosine_similarity(adj_flat.unsqueeze(1), adj_flat.unsqueeze(0), dim=-1)

        sim_matrix = ((adj_sim + ops_sim) / 2).to(preds.device)

        diff_matrix = (preds.unsqueeze(1) - preds.unsqueeze(0)) ** 2
        diff_matrix = diff_matrix.to(preds.device)

        structure_penalty = (sim_matrix * diff_matrix).mean()


        # 总损失
        total_loss = base_loss + self.lambda_struct * structure_penalty
        return total_loss


def normalize_adj(adj):
    last_dim = adj.size(-1)
    rowsum = adj.sum(1, keepdim=True)
    return torch.div(adj, rowsum)

def graph_pooling(inputs, num_vertices):
    out = inputs.sum(1)
    return torch.div(out, num_vertices.unsqueeze(-1).expand_as(out))

class DirectedMultiHopGCN_Enhanced(nn.Module):
    def __init__(self, in_features, out_features, k_hops=3):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.k_hops = k_hops

        # 每跳的正向、反向独立线性映射（W1/W2）
        self.forward_weights = nn.ModuleList([
            nn.Linear(in_features, out_features) for _ in range(k_hops)
        ])
        self.backward_weights = nn.ModuleList([
            nn.Linear(in_features, out_features) for _ in range(k_hops)
        ])

        # 每跳的交互融合模块
        self.fusion_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(out_features * 2, out_features),
                nn.ReLU()
            ) for _ in range(k_hops)
        ])

        # 每跳的注意力权重向量
        self.hop_attn = nn.Parameter(torch.randn(k_hops, out_features))

        # 最终融合与归一化
        self.norm = nn.LayerNorm(out_features)

        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.forward_weights + self.backward_weights:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        for mlp in self.fusion_mlps:
            for sublayer in mlp:
                if isinstance(sublayer, nn.Linear):
                    nn.init.xavier_uniform_(sublayer.weight)
                    nn.init.zeros_(sublayer.bias)
        nn.init.normal_(self.hop_attn, mean=0.0, std=0.1)

    def forward(self, x, adj):
        """
        x: [B, N, in_features]
        adj: [B, N, N]
        """
        B, N, _ = x.shape
        device = x.device
        identity = torch.eye(N, device=device).unsqueeze(0).expand(B, N, N)

        hop_features = []
        adj_power_fwd = identity
        adj_power_bwd = identity

        for k in range(self.k_hops):
            # 更新邻接矩阵幂
            adj_power_fwd = torch.bmm(adj_power_fwd, adj) if k > 0 else adj
            norm_adj_fwd = self.normalize_adj(adj_power_fwd)

            adj_power_bwd = torch.bmm(adj_power_bwd, adj.transpose(1, 2)) if k > 0 else adj.transpose(1, 2)
            norm_adj_bwd = self.normalize_adj(adj_power_bwd)

            # 正向、反向方向分离特征变换
            fwd_feat = self.forward_weights[k](x)
            fwd_feat = torch.bmm(norm_adj_fwd, fwd_feat)

            bwd_feat = self.backward_weights[k](x)
            bwd_feat = torch.bmm(norm_adj_bwd, bwd_feat)

            # 正反向融合
            concat_feat = torch.cat([fwd_feat, bwd_feat], dim=-1)
            fused = self.fusion_mlps[k](concat_feat)
            hop_features.append(fused)

        # 多跳特征堆叠：[B, k_hops, N, D]
        hop_stack = torch.stack(hop_features, dim=1)

        # 每跳注意力：[B, k_hops, N]
        attn_scores = (hop_stack * self.hop_attn.view(1, self.k_hops, 1, self.out_features)).sum(dim=-1)
        attn_scores = F.softmax(attn_scores, dim=1).unsqueeze(-1)  # [B, k_hops, N, 1]

        output = (hop_stack * attn_scores).sum(dim=1)  # [B, N, D]
        output = F.relu(self.norm(output))
        return output

    @staticmethod
    def normalize_adj(adj):
        """对称归一化邻接矩阵"""
        batch_size = adj.size(0)
        eye = torch.eye(adj.size(1), device=adj.device).unsqueeze(0).expand(batch_size, -1, -1)
        adj = adj + eye
        rowsum = adj.sum(dim=2, keepdim=True).clamp(min=1e-12)
        colsum = adj.sum(dim=1, keepdim=True).clamp(min=1e-12)
        norm_adj = adj / (torch.sqrt(rowsum) * torch.sqrt(colsum))
        return norm_adj

    def extra_repr(self):
        return f'in_features={self.in_features}, out_features={self.out_features}, k_hops={self.k_hops}'

class Predictor(nn.Module):
    def __init__(self, feature_dim):
        super(Predictor, self).__init__()
        self.fc = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, feature_dim),
            nn.PReLU(),
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, int(feature_dim / 2)),
            nn.PReLU(),
            nn.LayerNorm(int(feature_dim / 2)),
            nn.Linear(int(feature_dim / 2), int(feature_dim / 4)),
            nn.PReLU(),
            nn.LayerNorm(int(feature_dim / 4)),
            nn.Linear(int(feature_dim / 4), int(feature_dim / 8)),
            nn.PReLU(),
            nn.LayerNorm(int(feature_dim / 8)),
            nn.Linear(int(feature_dim / 8), int(feature_dim / 16)),
            nn.PReLU(),
            nn.LayerNorm(int(feature_dim / 16)),
            nn.Linear(int(feature_dim / 16), 1)
        )
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, z):
        return self.fc(z)

class MultimodalBottleneckTransformer(nn.Module):
    def __init__(self, feature_dim):
        super(MultimodalBottleneckTransformer, self).__init__()
        self.feature_dim = feature_dim
        self.pred_fc = Predictor(feature_dim)

    def forward(self, features):
        output = self.pred_fc(features)
        output = output.mean(dim=1)
        return output


class PredictorModel(nn.Module):
  def __init__(self, args, device):
    super(PredictorModel, self).__init__()
    self.hs = args.hs
    self.nz = args.nz
    self.device = device

    self.predictor = MultimodalBottleneckTransformer(1024)

    self.loss = BPRLoss()
    #self.loss = StructAwareBPRLoss()

    self.initial_hidden = 4
    self.gcn_hidden = self.nz
    self.gcn_layers = 6

    self.gcn = [DirectedMultiHopGCN_Enhanced(self.initial_hidden if i == 0 else self.gcn_hidden, self.gcn_hidden) for i in range(self.gcn_layers)]
    self.gcn = nn.ModuleList(self.gcn)

  def to_categorical(self, labels, num_classes, dtype='int8'):
      labels = labels.astype(int)
      one_hot = np.zeros((len(labels), num_classes), dtype=dtype)
      one_hot[np.arange(len(labels)), labels] = 1
      return one_hot

  def expand_adjacency_matrix(self, adj_matrix):
      current_size = adj_matrix.shape[0]

      if current_size < 7:
          expanded_matrix = np.zeros((7, 7), dtype=int)
          expanded_matrix[0:current_size, 0:current_size] = adj_matrix
          return expanded_matrix
      else:
          return adj_matrix

  def get_arch(self, important_metrics_batch, integers2one_hot=True):
      adjacency_batch = []
      operations_batch = []
      num_batch = []

      for metric in important_metrics_batch:
          fixed_metrics = metric['fixed_metrics']
          adjacent_matrix = fixed_metrics['module_adjacency']
          adjacent_matrix = self.expand_adjacency_matrix(adjacent_matrix)
          module_integers = fixed_metrics['module_integers']
          if len(module_integers) < 7:
              module_integers = np.append(module_integers, [0] * (7 - len(module_integers)))
          if integers2one_hot:
              module_integers = self.to_categorical(module_integers, 4, dtype='int8')

          num_vert = len(module_integers) - np.count_nonzero(module_integers == 0)

          adjacency_batch.append(adjacent_matrix)
          operations_batch.append(module_integers)
          num_batch.append(num_vert)

      adjacency_batch_np = np.array(adjacency_batch)
      adjacency_batch = torch.tensor(adjacency_batch_np)
      if isinstance(operations_batch, list):
          operations_batch = np.array(operations_batch)
          operations_batch = torch.tensor(operations_batch, dtype=torch.float32)
      num_batch = torch.tensor(num_batch)

      return num_batch, adjacency_batch, operations_batch

  def get_arch_features(self, num_vert_batch, adjacency_batch, operations_batch):
      numv, adj, out = num_vert_batch.to(self.device), adjacency_batch.to(self.device), operations_batch.to(self.device)
      numv = torch.full_like(numv, 7)
      adj_with_diag = normalize_adj(adj + torch.eye(7, device=adj.device))
      for layer in self.gcn:
          out = layer(out, adj_with_diag)
      out = graph_pooling(out, numv)

      return out

