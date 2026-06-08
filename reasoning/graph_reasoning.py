"""
时序图神经网络推理模块
=========================
核心算法:
1. 公式(13): Score_graph(eo) = ⟨fg(q), eo⟩
   图神经网络通过时间感知邻居聚合编码查询和实体

实现基于TiRGN/RE-GCN风格的时序图神经网络:
- 实体/关系嵌入层
- 时间感知邻居聚合
- 时序衰减注意力
- 查询编码与候选实体评分
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Set, Optional
from collections import defaultdict

import os

from ..config import (
    EMBEDDING_DIM, GNN_LAYERS, GNN_DROPOUT,
    GNN_LR, GNN_EPOCHS, GNN_BATCH_SIZE, LAMBDA_DECAY, DEVICE,
    RETRAIN_MODEL, GNN_MODEL_DIR,
)
from ..data.dataset import TKGDataSet, Quadruple


class TimeAwareGNNLayer(nn.Module):
    """
    时间感知图神经网络层
    时间感知邻居聚合

    对每个实体的邻居按时间衰减权重聚合特征
    """

    def __init__(self, dim: int, dropout: float = 0.2):
        super().__init__()
        self.dim = dim
        self.msg_fc = nn.Linear(dim * 3, dim)       # 消息函数
        self.update_fc = nn.Linear(dim * 2, dim)     # 更新函数
        self.time_fc = nn.Linear(1, dim)             # 时间编码
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim)

    def forward(self,
                entity_emb: torch.Tensor,
                rel_emb: torch.Tensor,
                adj_triples: List[Tuple[int, int, int, float]],
                num_entities: int) -> torch.Tensor:
        """
        Args:
            entity_emb: [num_entities, dim]
            rel_emb: [num_relations, dim]
            adj_triples: [(head_idx, rel_idx, tail_idx, time_val), ...]
            num_entities: 用于确保输出维度
        """
        if not adj_triples:
            return entity_emb

        # 按头实体聚合邻居消息
        neighbor_msgs = defaultdict(list)

        for h, r, t, time_val in adj_triples:
            head_vec = entity_emb[h]
            rel_vec = rel_emb[r]
            tail_vec = entity_emb[t]

            # 时间特征编码
            time_feat = self.time_fc(
                torch.tensor([[time_val]], dtype=torch.float32, device=entity_emb.device)
            )

            # 消息 = f(head, relation, tail) + time_encoding
            msg_input = torch.cat([head_vec, rel_vec, tail_vec])
            msg = self.msg_fc(msg_input) + time_feat.squeeze(0)
            neighbor_msgs[t].append(msg)  # 聚合到tail实体

        # 更新实体表示
        updated_emb = entity_emb.clone()
        for entity_idx, msgs in neighbor_msgs.items():
            if entity_idx >= num_entities:
                continue
            if msgs:
                agg_msg = torch.stack(msgs).mean(dim=0)
                # 更新: h' = LayerNorm(h + Dropout(W[h; agg]))
                combined = torch.cat([entity_emb[entity_idx], agg_msg])
                update = self.update_fc(combined)
                updated_emb[entity_idx] = self.layer_norm(
                    entity_emb[entity_idx] + self.dropout(update)
                )

        return updated_emb


class TemporalGNN(nn.Module):
    """
    时序图神经网络 (类似TiRGN/RE-GCN架构)
    时间感知图推理
    """

    def __init__(self,
                 num_entities: int,
                 num_relations: int,
                 dim: int = 128,
                 num_layers: int = 2,
                 dropout: float = 0.2):
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.dim = dim

        # 嵌入层
        self.entity_emb = nn.Embedding(num_entities, dim)
        self.relation_emb = nn.Embedding(num_relations, dim)
        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.xavier_uniform_(self.relation_emb.weight)

        # 时间感知GNN层
        self.gnn_layers = nn.ModuleList([
            TimeAwareGNNLayer(dim, dropout)
            for _ in range(num_layers)
        ])

        # 查询编码器: fg(q) = W[es_emb; rq_emb; time_emb]
        self.query_encoder = nn.Sequential(
            nn.Linear(dim * 3, dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )

        self.time_encoder = nn.Linear(1, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self,
                adj_triples: List[Tuple[int, int, int, float]],
                query_entity_idx: int,
                query_rel_idx: int,
                query_time: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播

        Returns:
            (query_encoding, all_entity_embeddings)
        """
        entity_emb = self.entity_emb.weight
        rel_emb = self.relation_emb.weight

        # 逐层GNN传播
        for gnn_layer in self.gnn_layers:
            entity_emb = gnn_layer(entity_emb, rel_emb, adj_triples, self.num_entities)

        # 查询编码: fg(q) = W[es || rq || t_emb]
        es_emb = entity_emb[query_entity_idx]
        rq_emb = rel_emb[query_rel_idx]
        t_feat = self.time_encoder(
            torch.tensor([[query_time]], dtype=torch.float32, device=entity_emb.device)
        ).squeeze(0)

        query_vec = torch.cat([es_emb, rq_emb, t_feat])
        query_encoding = self.query_encoder(query_vec)

        return query_encoding, entity_emb

    # --------------------------------------------------------
    # Score_graph(eo) = ⟨fg(q), eo⟩
    # --------------------------------------------------------
    def score_entities(self,
                       query_encoding: torch.Tensor,
                       entity_embeddings: torch.Tensor) -> torch.Tensor:
        """
        论文公式(13): Score_graph(eo) = ⟨fg(q), eo⟩

        计算查询编码与所有实体向量的内积作为得分

        Returns:
            [num_entities] 得分向量
        """
        # 归一化以提高内积稳定性
        q_norm = F.normalize(query_encoding, dim=-1)
        e_norm = F.normalize(entity_embeddings, dim=-1)
        scores = torch.matmul(e_norm, q_norm)
        return scores


class GraphReasoner:
    """
    图推理器
    封装GNN的训练和推理
    """

    def __init__(self, dataset: TKGDataSet):
        self.dataset = dataset

        # 构建时序邻接列表 (用于GNN)
        self.adj_triples = self._build_adjacency()
        self.time_sorted_adj = sorted(self.adj_triples, key=lambda x: x[3])

        self.model = TemporalGNN(
            num_entities=dataset.num_entities,
            num_relations=dataset.num_relations,
            dim=EMBEDDING_DIM,
            num_layers=GNN_LAYERS,
            dropout=GNN_DROPOUT,
        ).to(DEVICE)

        self.is_trained = False
        os.makedirs(GNN_MODEL_DIR, exist_ok=True)
        self.model_path = os.path.join(GNN_MODEL_DIR, f"gnn_{dataset.name}.pt")

    def _build_adjacency(self) -> List[Tuple[int, int, int, float]]:
        """构建格式化的邻接列表"""
        adj = []
        for q in self.dataset.quadruples:
            s_idx = self.dataset.entity2id.get(q.subject)
            r_idx = self.dataset.relation2id.get(q.relation)
            o_idx = self.dataset.entity2id.get(q.object)
            if all(x is not None for x in [s_idx, r_idx, o_idx]):
                adj.append((s_idx, r_idx, o_idx, q.time_val))
        return adj

    def get_adj_before_time(self, query_time: float) -> List:
        """获取查询时间前的邻接列表 (仅使用历史数据)"""
        return [t for t in self.time_sorted_adj if t[3] <= query_time]

    def train_model(self, epochs: int = None, lr: float = None):
        """
        训练GNN模型

        使用对比学习风格: 正样本(实际存在的四元组)得分应高于负样本
        """
        if epochs is None:
            epochs = GNN_EPOCHS
        if lr is None:
            lr = GNN_LR

        print(f"\n[GNN训练]  - 时序图神经网络")
        print(f"  实体={self.dataset.num_entities}, 关系={self.dataset.num_relations}")
        print(f"  嵌入维度={EMBEDDING_DIM}, 层数={GNN_LAYERS}, Epochs={epochs}")

        # 尝试加载缓存模型
        if not RETRAIN_MODEL and os.path.exists(self.model_path):
            try:
                self.model.load_state_dict(
                    torch.load(self.model_path, map_location=DEVICE)
                )
                self.model.to(DEVICE)
                self.is_trained = True
                print(f"  [GNN] 已加载缓存模型: {self.model_path}")
                return
            except Exception as e:
                print(f"  [GNN] 加载缓存失败 ({e}), 重新训练")

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        self.model.train()

        train_triples = [(self.dataset.entity2id[q.subject],
                          self.dataset.relation2id[q.relation],
                          self.dataset.entity2id[q.object],
                          q.time_val)
                         for q in self.dataset.train_quads
                         if q.subject in self.dataset.entity2id
                         and q.relation in self.dataset.relation2id
                         and q.object in self.dataset.entity2id]

        num_batches = max(1, len(train_triples) // GNN_BATCH_SIZE)
        # 每个 batch 内采样的 triples 数量 (自适应: 小数据集多采, 大数据集少采)
        samples_per_batch = min(32, max(8, 256 // num_batches))

        for epoch in range(epochs):
            total_loss = 0.0
            np.random.shuffle(train_triples)

            for b in range(num_batches):
                batch = train_triples[b * GNN_BATCH_SIZE:(b + 1) * GNN_BATCH_SIZE]
                if not batch:
                    continue

                # 构建该批次的邻接子图
                batch_adj = [(h, r, t, time_v) for h, r, t, time_v in batch]

                batch_loss = 0.0
                for h, r, t_true, time_v in batch[:samples_per_batch]:
                    hist_adj = [a for a in batch_adj if a[3] <= time_v]
                    if not hist_adj:
                        hist_adj = batch_adj[:1]

                    query_enc, entity_emb = self.model(hist_adj, h, r, time_v)
                    scores = self.model.score_entities(query_enc, entity_emb)

                    pos_score = scores[t_true]

                    neg_entities = np.random.choice(
                        self.dataset.num_entities,
                        size=min(10, self.dataset.num_entities - 1),
                        replace=False
                    )
                    neg_entities = [n for n in neg_entities if n != t_true]

                    if neg_entities:
                        neg_scores = scores[neg_entities]
                        margin = 0.5
                        neg_loss = F.relu(margin - pos_score + neg_scores).mean()
                        batch_loss += neg_loss

                if batch_loss > 0:
                    optimizer.zero_grad()
                    batch_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                    total_loss += batch_loss.item()

            if (epoch + 1) % max(1, epochs // 5) == 0:
                avg_loss = total_loss / max(1, num_batches)
                print(f"  Epoch {epoch + 1}/{epochs}, Loss={avg_loss:.4f}")

        self.is_trained = True
        try:
            torch.save(self.model.state_dict(), self.model_path)
            print(f"  GNN模型已保存: {self.model_path}")
        except Exception as e:
            print(f"  GNN模型保存失败: {e}")
        print(f"  GNN训练完成!")

    def reason(self,
               query_entity: str,
               query_relation: str,
               query_time: str,
               auto_train: bool = True) -> Dict[str, float]:
        """
        执行图推理

        论文公式(13): Score_graph(eo) = ⟨fg(q), eo⟩

        Returns:
            Dict[entity, score]: 候选实体及其图推理得分
        """
        if not self.is_trained and auto_train:
            print("[GNN] 模型未训练, 开始训练...")
            self.train_model()

        self.model.eval()

        s_idx = self.dataset.entity2id.get(query_entity)
        r_idx = self.dataset.relation2id.get(query_relation)
        tq = Quadruple._parse_time(query_time)

        if s_idx is None or r_idx is None:
            return {}

        with torch.no_grad():
            # 使用查询时间前的邻接
            hist_adj = self.get_adj_before_time(tq)
            if not hist_adj:
                hist_adj = self.adj_triples[:100]

            query_enc, entity_emb = self.model(hist_adj, s_idx, r_idx, tq)
            scores = self.model.score_entities(query_enc, entity_emb)

            # 转换为字典
            entity_scores = {}
            for idx in range(self.dataset.num_entities):
                entity_name = self.dataset.id2entity.get(idx)
                if entity_name and entity_name != query_entity:
                    score_val = float(scores[idx].item())
                    if score_val > 0:  # 仅保留正相关
                        entity_scores[entity_name] = score_val

        return entity_scores
