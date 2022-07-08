# -*- coding: UTF-8 -*-
'''
DAGN Version 2.1.3.2

Adapted from: https://github.com/llamazing/numnet_plus
Date: 8/11/2020
Author: Yinya Huang
'''

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import numpy as np
from typing import List, Dict, Any, Tuple
from itertools import groupby
from operator import itemgetter
import copy
from util import FFNLayer, ResidualGRU, ArgumentGCN, ArgumentGCN_wreverseedges_double
from tools import allennlp as util
from transformers import BertPreTrainedModel, RobertaModel, BertModel
import torch.nn.functional as F


def load_model():
    # lm_model = RobertaForMaskedLM.from_pretrained('roberta-base').cuda()
    lm_model = torch.load("/data/linqika/xufangzhi/parallel/test/temp_check/reclor_spanmask_1-10/reclor_4.pth")
    mc_model = RobertaModel.from_pretrained('roberta-large')

    mc_model.roberta.embeddings.load_state_dict(lm_model.roberta.embeddings.state_dict())
    mc_model.roberta.encoder.load_state_dict(lm_model.roberta.encoder.state_dict())

    return mc_model


class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(hidden_size, ffn_size)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(ffn_size, hidden_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, attention_dropout_rate, num_heads):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads

        self.att_size = att_size = hidden_size // num_heads
        self.scale = att_size ** -0.5

        self.linear_q = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_k = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_v = nn.Linear(hidden_size, num_heads * att_size)
        self.att_dropout = nn.Dropout(attention_dropout_rate)

        self.output_layer = nn.Linear(num_heads * att_size, hidden_size)

        self.attn_bias_linear = nn.Linear(1, self.num_heads)

    def forward(self, q, k, v, attn_bias=None, attention_mask=None):
        orig_q_size = q.size()

        d_k = self.att_size
        d_v = self.att_size
        batch_size = q.size(0)

        # head_i = Attention(Q(W^Q)_i, K(W^K)_i, V(W^V)_i)
        q = self.linear_q(q).view(batch_size, -1, self.num_heads, d_k)
        k = self.linear_k(k).view(batch_size, -1, self.num_heads, d_k)
        v = self.linear_v(v).view(batch_size, -1, self.num_heads, d_v)

        q = q.transpose(1, 2)  # [b, h, q_len, d_k]
        v = v.transpose(1, 2)  # [b, h, v_len, d_v]
        k = k.transpose(1, 2).transpose(2, 3)  # [b, h, d_k, k_len]

        # Scaled Dot-Product Attention.
        # Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k))V
        q = q * self.scale
        x = torch.matmul(q, k)  # [b, h, q_len, k_len]
        if attn_bias is not None:
            attn_bias = attn_bias.unsqueeze(-1).permute(0, 3, 1, 2)
            attn_bias = attn_bias.repeat(1, self.num_heads, 1, 1)
            x += attn_bias

        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(-1).permute(0, 3, 1, 2)
            attention_mask = attention_mask.repeat(1, self.num_heads, 1, 1)
            x += attention_mask

        x = torch.softmax(x, dim=3)
        x = self.att_dropout(x)
        x = x.matmul(v)  # [b, h, q_len, attn]

        x = x.transpose(1, 2).contiguous()  # [b, q_len, h, attn]
        x = x.view(batch_size, -1, self.num_heads * d_v)

        x = self.output_layer(x)

        assert x.size() == orig_q_size
        return x


class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate, attention_dropout_rate, num_heads, attn_bias=None):
        super(EncoderLayer, self).__init__()

        self.self_attention_norm = nn.LayerNorm(hidden_size)
        self.self_attention = MultiHeadAttention(
            hidden_size, attention_dropout_rate, num_heads)
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def forward(self, x, attn_bias=None, attention_mask=None):
        y = self.self_attention_norm(x)
        y = self.self_attention(y, y, y, attn_bias, attention_mask)
        y = self.self_attention_dropout(y)
        x = x + y

        y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y
        return x


class Position_Embedding(nn.Module):
    def __init__(self, hidden_size):
        super(Position_Embedding, self).__init__()
        self.hidden_size = hidden_size

    def forward(self, x):  # input is encoded spans
        batch_size = x.size(0)
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long)
        pos = pos.unsqueeze(0).expand(batch_size, seq_len)  # [seq_len] -> [batch_size, seq_len]
        self.pos_embed = nn.Embedding(seq_len, self.hidden_size)  # position embedding
        embedding = self.pos_embed(pos)

        return embedding.to(x.device)


class Link_Importance_Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Link_Importance_Attention, self).__init__()
        self.Linear_1 = nn.Linear(hidden_size, hidden_size)
        self.Linear_2 = nn.Linear(hidden_size, hidden_size)

    def tensor_norm(self, tensor):
        mean = tensor.mean(dim=-1)
        std = tensor.std(dim=-1)
        return (tensor - mean) / std

    def forward(self, encoded_spans):
        batch_size = encoded_spans.size(0)
        seq_len = encoded_spans.size(1)

        # Previous Methods for computing the link importance with hadamard product
        encoded_spans = torch.softmax(encoded_spans, dim=-1)
        link_attn = torch.matmul(self.Linear_1(encoded_spans),
                                 self.Linear_2(encoded_spans).permute(0, 2, 1))  # --> batch_size*N*N
        link_attn = torch.softmax(link_attn, dim=-1)

        return link_attn.to(encoded_spans.device)


class Question_Aware_Weights(nn.Module):
    def __init__(self, hidden_size):
        super(Question_Aware_Weights, self).__init__()
        self.FF_layer1 = nn.Linear(hidden_size, hidden_size // 2)
        self.gelu = nn.GELU()
        self.FF_layer2 = nn.Linear(hidden_size // 2, 2)

    def forward(self, q, encoded_spans):
        all_node = encoded_spans.mean(dim=1)
        x = q + all_node
        x = self.FF_layer1(x)
        x = self.gelu(x)
        x = self.FF_layer2(x)
        x = torch.softmax(x, dim=-1)
        return x


class DAGN_ques(BertPreTrainedModel):
    '''
    Adapted from https://github.com/llamazing/numnet_plus.

    Inputs of forward(): see try_data_5.py - the outputs of arg_tokenizer()
        - input_ids: list[int]
        - attention_mask: list[int]
        - segment_ids: list[int]
        - argument_bpe_ids: list[int]. value={ -1: padding,
                                                0: non_arg_non_dom,
                                                1: (relation, head, tail)  关键词在句首
                                                2: (head, relation, tail)  关键词在句中，先因后果
                                                3: (tail, relation, head)  关键词在句中，先果后因
                                                }
        - domain_bpe_ids: list[int]. value={ -1: padding,
                                              0:non_arg_non_dom,
                                           D_id: domain word ids.}
        - punctuation_bpe_ids: list[int]. value={ -1: padding,
                                                   0: non_punctuation,
                                                   1: punctuation}


    '''

    def __init__(self,
                 config,
                 init_weights: bool,
                 max_rel_id,
                 hidden_size: int,
                 dropout_prob: float = 0.1,
                 merge_type: int = 1,
                 token_encoder_type: str = "roberta",
                 gnn_version: str = "GCN",
                 use_pool: bool = False,
                 use_gcn: bool = False,
                 gcn_steps: int = 1) -> None:
        super().__init__(config)

        self.layer_num = 5
        self.head_num = 5
        self.token_encoder_type = token_encoder_type
        self.max_rel_id = max_rel_id
        self.merge_type = merge_type
        self.use_gcn = use_gcn
        self.use_pool = use_pool
        assert self.use_gcn or self.use_pool

        ''' from modeling_roberta '''
        self.roberta = RobertaModel(config)
        # self.roberta = RobertaModel.from_pretrained("/data/linqika/xufangzhi/parallel/test/temp_check/reclor_logiqa_spanmask_1-10/")
        if self.use_pool:
            self.dropout = nn.Dropout(config.hidden_dropout_prob)
            self.classifier = nn.Linear(config.hidden_size, 1)

        ''' from numnet '''
        if self.use_gcn:
            modeling_out_dim = hidden_size
            node_dim = modeling_out_dim

            self._gcn_input_proj = nn.Linear(node_dim * 2, node_dim)
            if gnn_version == "GCN":
                self._gcn = ArgumentGCN(node_dim=node_dim, iteration_steps=gcn_steps)
            elif gnn_version == "GCN_reversededges_double":
                self._gcn = ArgumentGCN_wreverseedges_double(node_dim=node_dim, iteration_steps=gcn_steps)
            else:
                print("gnn_version == {}".format(gnn_version))
                raise Exception()
            self._iteration_steps = gcn_steps
            self._gcn_prj_ln = nn.LayerNorm(node_dim)
            # self._gcn_enc = ResidualGRU(hidden_size, 0, 2)
            self._gcn_enc = nn.Sequential(nn.Linear(1024, 1024, bias=False), nn.ReLU())
            self._proj_sequence_h = nn.Linear(hidden_size, 1, bias=False)

            # span num extraction
            self._proj_span_num = FFNLayer(3 * hidden_size, hidden_size, 1, dropout_prob)

            self._proj_gcn_pool = FFNLayer(3 * hidden_size, hidden_size, 1, dropout_prob)
            self._proj_gcn_pool_4 = FFNLayer(4 * hidden_size, hidden_size, 1, dropout_prob)
            self._proj_gcn_pool_3 = FFNLayer(2 * hidden_size, hidden_size, 1, dropout_prob)

            self.pre_ln = nn.LayerNorm(hidden_size)
            self.embed_to_value = nn.Sequential(nn.Linear(1024, 1, bias=False), nn.ReLU())

            self.pos_embed = Position_Embedding(hidden_size)
            self.link_importance = Link_Importance_Attention(hidden_size)
            self.question_aware_weights = Question_Aware_Weights(hidden_size)

            self.input_dropout = nn.Dropout(dropout_prob)
            encoders = [EncoderLayer(hidden_size, hidden_size, dropout_prob, dropout_prob, self.head_num)
                        for _ in range(self.layer_num)]
            self.encoder_layers = nn.ModuleList(encoders)

            self.final_ln = nn.LayerNorm(hidden_size)

        if init_weights:
            self.init_weights()

    def split_into_spans_9(self, seq, seq_mask, split_bpe_ids, passage_mask, option_mask, question_mask):
        '''

            :param seq: (bsz, seq_length, embed_size)
            :param seq_mask: (bsz, seq_length)
            :param split_bpe_ids: (bsz, seq_length). value = {-1, 0, 1, 2, 3, 4}.
            :return:
                - encoded_spans: (bsz, n_nodes, embed_size)
                - span_masks: (bsz, n_nodes)
                - edges: (bsz, n_nodes - 1)
                - node_in_seq_indices: list of list of list(len of span).

        '''

        def _consecutive(seq: list, vals: np.array):
            groups_seq = []
            output_vals = copy.deepcopy(vals)
            for k, g in groupby(enumerate(seq), lambda x: x[0] - x[1]):
                groups_seq.append(list(map(itemgetter(1), g)))
            output_seq = []
            for i, ids in enumerate(groups_seq):
                output_seq.append(ids[0])
                if len(ids) > 1:
                    output_vals[ids[0]:ids[-1] + 1] = min(output_vals[ids[0]:ids[-1] + 1])
            return groups_seq, output_seq, output_vals

        embed_size = seq.size(-1)
        device = seq.device
        encoded_spans = []
        span_masks = []
        edges = []
        edges_embed = []
        node_in_seq_indices = []
        ques_seq = []
        for item_seq_mask, item_seq, item_split_ids, p_mask, o_mask, q_mask in zip(seq_mask, seq, split_bpe_ids,
                                                                                   passage_mask, option_mask,
                                                                                   question_mask):
            # item_seq_len = item_seq_mask.sum().item()
            item_seq_len = p_mask.sum().item() + o_mask.sum().item()  # item_seq = passage + option
            item_ques_seq = item_seq[item_seq_len:item_seq_mask.sum().item()]
            item_ques_seq = item_ques_seq.mean(dim=0)
            item_seq = item_seq[:item_seq_len]
            item_split_ids = item_split_ids[:item_seq_len]
            item_split_ids = item_split_ids.cpu().numpy()
            split_ids_indices = np.where(item_split_ids > 0)[0].tolist()
            grouped_split_ids_indices, split_ids_indices, item_split_ids = _consecutive(
                split_ids_indices, item_split_ids)
            # print(grouped_split_ids_indices)     [[0], [3], [14, 15, 16], [23], [28], [32], [34], [46], [58], [66, 67], [71], [81], [101, 102]]
            # print(split_ids_indices)   [0, 3, 14, 23, 28, 32, 34, 46, 58, 66, 71, 81, 101]
            # print(item_split_ids)
            # [5 0 0 5 0 0 0 0 0 0 0 0 0 0 4 4 4 0 0 0 0 0 0 5 0 0 0 0 5 0 0 0 5 0 4 0 0
            #    0 0 0 0 0 0 0 0 0 5 0 0 0 0 0 0 0 0 0 0 0 5 0 0 0 0 0 0 0 5 5 0 0 0 4 0 0
            #    0 0 0 0 0 0 0 4 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 5 5]

            n_split_ids = len(split_ids_indices)

            item_spans, item_mask = [], []
            item_edges = []
            item_edges_embed = []
            item_node_in_seq_indices = []
            item_edges.append(item_split_ids[split_ids_indices[0]])
            for i in range(n_split_ids):
                if i == n_split_ids - 1:
                    span = item_seq[split_ids_indices[i] + 1:]
                    if not len(span) == 0:
                        item_spans.append(span.sum(0))
                        item_mask.append(1)

                else:
                    span = item_seq[split_ids_indices[i] + 1:split_ids_indices[i + 1]]
                    if not len(span) == 0:
                        item_spans.append(span.sum(
                            0))  # span.sum(0) calculate the sum of embedding value at each position (1024 in total)
                        item_mask.append(1)
                        item_edges.append(item_split_ids[split_ids_indices[i + 1]])  # the edge type after the span
                        item_edges_embed.append(item_seq[split_ids_indices[i + 1]])  # the edge embedding after the span
                        item_node_in_seq_indices.append([i for i in range(grouped_split_ids_indices[i][-1] + 1,
                                                                          grouped_split_ids_indices[i + 1][
                                                                              0])])  # node indices [[1, 2], [4, 5, 6, 7, 8, 9, 10, 11, 12, 13]....]
            encoded_spans.append(item_spans)
            span_masks.append(item_mask)
            edges.append(item_edges)
            edges_embed.append(item_edges_embed)
            node_in_seq_indices.append(item_node_in_seq_indices)
            ques_seq.append(item_ques_seq)

        max_nodes = max(map(len, span_masks))  # span_masks:[n_choice * batch_size, node_num]
        span_masks = [spans + [0] * (max_nodes - len(spans)) for spans in
                      span_masks]  # make the node number be the same
        span_masks = torch.from_numpy(np.array(span_masks))
        span_masks = span_masks.to(device).long()

        pad_embed = torch.zeros(embed_size, dtype=seq.dtype, device=seq.device)
        attention_mask = torch.zeros((seq.size(0), max_nodes, max_nodes), dtype=seq.dtype, device=seq.device)
        attention_mask += -1e9
        for i, spans in enumerate(encoded_spans):
            attention_mask[i, :, :len(spans)] = 0

        encoded_spans = [spans + [pad_embed] * (max_nodes - len(spans)) for spans in
                         encoded_spans]  # [n_choice * batch_size, max_node_num, hidden_size]
        encoded_spans = [torch.stack(lst, dim=0) for lst in encoded_spans]
        encoded_spans = torch.stack(encoded_spans, dim=0)
        encoded_spans = encoded_spans.to(device).float()  # encoded_spans: (bsz x n_choices, n_nodes, embed_size)
        # Truncate head and tail of each list in edges HERE.
        #     Because the head and tail edge DO NOT contribute to the argument graph and punctuation graph.
        truncated_edges = [item[1:-1] for item in edges]
        truncated_edges_embed = [item[1:-1] for item in edges_embed]
        ques_seq = torch.stack(ques_seq, dim=0)

        return encoded_spans, span_masks, truncated_edges, truncated_edges_embed, node_in_seq_indices, attention_mask, ques_seq

    def get_gcn_info_vector(self, indices, node, size, device):
        '''
        give the node embed to each token in one node

        :param indices: list(len=bsz) of list(len=n_notes) of list(len=varied).
        :param node: (bsz, n_nodes, embed_size)
        :param size: value=(bsz, seq_len, embed_size)
        :param device:
        :return:
        '''

        batch_size = size[0]
        gcn_info_vec = torch.zeros(size=size, dtype=torch.float, device=device)

        for b in range(batch_size):
            for ids, emb in zip(indices[b], node[b]):
                gcn_info_vec[b, ids] = emb

        return gcn_info_vec

    def get_adjacency_matrices_2(self, edges: List[List[int]], edges_embed: List[List[int]], n_nodes: int,
                                 device: torch.device):
        '''
        Convert the edge_value_list into adjacency matrices.
            * argument graph adjacency matrix. Asymmetric (directed graph).
            * punctuation graph adjacency matrix. Symmetric (undirected graph).

            : argument
                - edges:list[list[str]]. len_out=(bsz x n_choices), len_in=n_edges. value={-1, 0, 1, 2, 3, 4, 5}.

            Note: relation patterns
                1 - (relation, head, tail)  关键词在句首
                2 - (head, relation, tail)  关键词在句中，先因后果
                3 - (tail, relation, head)  关键词在句中，先果后因
                4 - (head, relation, tail) & (tail, relation, head)  (1) argument words 中的一些关系
                5 - (head, relation, tail) & (tail, relation, head)  (2) punctuations

        '''
        batch_size = len(edges)
        hidden_size = 1024
        argument_graph = torch.zeros(
            (batch_size, n_nodes, n_nodes))  # NOTE: the diagonal should be assigned 0 since is acyclic graph.
        punct_graph = torch.zeros(
            (batch_size, n_nodes, n_nodes))  # NOTE: the diagonal should be assigned 0 since is acyclic graph.
        edges_embed_graph = torch.zeros(
            (batch_size, n_nodes, n_nodes,
             hidden_size))  # NOTE: the diagonal should be assigned 0 since is acyclic graph.
        edges_value_graph = torch.zeros(
            (batch_size, n_nodes, n_nodes))  # NOTE: the diagonal should be assigned 0 since is acyclic graph.
        for b, sample_edges in enumerate(edges):
            for i, edge_value in enumerate(sample_edges):
                if edge_value == 1:  # (relation, head, tail)  关键词在句首. Note: not used in graph_version==4.0.
                    try:
                        argument_graph[b, i + 1, i + 2] = 1
                        edges_value_graph[b, i + 1, i + 2] = 1
                    except Exception:
                        pass
                elif edge_value == 2:  # (head, relation, tail)  关键词在句中，先因后果. Note: not used in graph_version==4.0.
                    argument_graph[b, i, i + 1] = 1
                    edges_value_graph[b, i, i + 1] = 2
                elif edge_value == 3:  # (tail, relation, head)  关键词在句中，先果后因. Note: not used in graph_version==4.0.
                    argument_graph[b, i + 1, i] = 1
                    edges_value_graph[b, i + 1, i] = 3
                elif edge_value == 4:  # (head, relation, tail) & (tail, relation, head) ON ARGUMENT GRAPH
                    argument_graph[b, i, i + 1] = 1
                    argument_graph[b, i + 1, i] = 1
                    edges_value_graph[b, i, i + 1] = 4
                    edges_value_graph[b, i + 1, i] = 4
                elif edge_value == 5:  # (head, relation, tail) & (tail, relation, head) ON PUNCTUATION GRAPH
                    try:
                        punct_graph[b, i, i + 1] = 1
                        punct_graph[b, i + 1, i] = 1
                        edges_value_graph[b, i, i + 1] = 5
                        edges_value_graph[b, i + 1, i] = 5
                    except Exception:
                        pass

        for b, sample_edges_embed in enumerate(edges_embed):  # iter edges_embed
            for i, edge_embed_value in enumerate(sample_edges_embed):
                edges_embed_graph[b, i, i + 1, :] = torch.as_tensor(edge_embed_value).to(device)
                edges_embed_graph[b, i + 1, i, :] = torch.as_tensor(edge_embed_value).to(device)

        return argument_graph.to(device), punct_graph.to(device), edges_embed_graph.to(device), edges_value_graph.to(
            device)

    def compute_distance(self, sequence_output, seq_mask, passage_mask, option_mask, question_mask, logits):
        '''
        compute the distance between context + question and option
        '''

        def tensor_norm(tensor):
            mean = tensor.mean(dim=-1)
            std = tensor.std(dim=-1)
            return (tensor - mean) / std

        preds = torch.argmax(logits, dim=-1)  # the predicted labels

        distance_loss = []
        for index, (item_seq, item_seq_mask, p_mask, o_mask, q_mask, pred) in enumerate(
                zip(sequence_output, seq_mask, passage_mask, option_mask, question_mask, preds)):
            if index % 4 == pred.item():
                item_passage_len = p_mask.sum().item()
                item_passage_option_len = p_mask.sum().item() + o_mask.sum().item()
                item_seq_len = item_seq_mask.sum().item()

                item_passage_seq = item_seq[:item_passage_len]
                item_option_seq = item_seq[item_passage_len:item_passage_option_len]
                item_ques_seq = item_seq[item_passage_option_len:item_seq_len]

                item_passage_seq = tensor_norm(item_passage_seq.mean(dim=0))
                item_option_seq = tensor_norm(item_option_seq.mean(dim=0))
                item_ques_seq = tensor_norm(item_ques_seq.mean(dim=0))

                dist = torch.dist(item_passage_seq + item_option_seq, item_ques_seq, p=1) / 1024  # 1范数
                distance_loss.append(dist)

        distance_loss = torch.tensor(distance_loss, requires_grad=True).mean()
        return distance_loss.to(sequence_output.device)

    def forward(self,
                input_ids: torch.LongTensor,
                attention_mask: torch.LongTensor,

                passage_mask: torch.LongTensor,
                option_mask: torch.LongTensor,
                question_mask: torch.LongTensor,

                argument_bpe_ids: torch.LongTensor,
                domain_bpe_ids: torch.LongTensor,
                punct_bpe_ids: torch.LongTensor,

                labels: torch.LongTensor,
                token_type_ids: torch.LongTensor = None,
                ) -> Tuple:

        num_choices = input_ids.shape[1]

        flat_input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None

        flat_passage_mask = passage_mask.view(-1, passage_mask.size(
            -1)) if passage_mask is not None else None  # [num_choice*batchsize, hidden_size]
        flat_option_mask = option_mask.view(-1, option_mask.size(
            -1)) if option_mask is not None else None  # [num_choice*batchsize, hidden_size]
        flat_question_mask = question_mask.view(-1, question_mask.size(
            -1)) if question_mask is not None else None  # [num_choice*batchsize, hidden_size]

        flat_argument_bpe_ids = argument_bpe_ids.view(-1, argument_bpe_ids.size(
            -1)) if argument_bpe_ids is not None else None
        flat_domain_bpe_ids = domain_bpe_ids.view(-1, domain_bpe_ids.size(-1)) if domain_bpe_ids is not None else None
        flat_punct_bpe_ids = punct_bpe_ids.view(-1, punct_bpe_ids.size(-1)) if punct_bpe_ids is not None else None

        bert_outputs = self.roberta(flat_input_ids, attention_mask=flat_attention_mask, token_type_ids=None)
        sequence_output = bert_outputs[0]
        pooled_output = bert_outputs[1]  # [bz*n_choice, hidden_size]

        if self.use_gcn:
            ''' The GCN branch. Suppose to go back to baseline once remove. '''
            new_punct_id = self.max_rel_id + 1
            new_punct_bpe_ids = new_punct_id * flat_punct_bpe_ids  # punct_id: 1 -> 4. for incorporating with argument_bpe_ids.
            _flat_all_bpe_ids = flat_argument_bpe_ids + new_punct_bpe_ids  # -1:padding, 0:non, 1-3: arg, 4:punct.
            overlapped_punct_argument_mask = (_flat_all_bpe_ids > new_punct_id).long()
            flat_all_bpe_ids = _flat_all_bpe_ids * (
                        1 - overlapped_punct_argument_mask) + flat_argument_bpe_ids * overlapped_punct_argument_mask
            assert flat_argument_bpe_ids.max().item() <= new_punct_id

            # encoded_spans: (bsz x n_choices, n_nodes, embed_size)
            # span_mask: (bsz x n_choices, n_nodes)
            # edges: list[list[int]]
            # node_in_seq_indices: list[list[list[int]]]

            encoded_spans, span_mask, edges, edges_embed, node_in_seq_indices, attention_mask, ques_seq = self.split_into_spans_9(
                sequence_output,
                flat_attention_mask,
                flat_all_bpe_ids,
                flat_passage_mask,
                flat_option_mask,
                flat_question_mask)
            argument_graph, punctuation_graph, edges_embed_graph, edges_value_graph = self.get_adjacency_matrices_2(
                edges, edges_embed=edges_embed, n_nodes=encoded_spans.size(1), device=encoded_spans.device)

            encoded_spans = encoded_spans + self.pos_embed(encoded_spans)  # node_embedding + positional embedding

            # encoded_spans = self.pre_ln(encoded_spans)
            node = self.input_dropout(encoded_spans)

            for enc_layer in self.encoder_layers:
                # weights = self.question_aware_weights(ques_seq, encoded_spans)
                # attn_bias = argument_graph
                # edges_embed_bias = self.embed_to_value(edges_embed_graph).squeeze(-1)
                # attn_bias = edges_embed_bias
                # attn_bias = edges_value_graph
                link_attn_bias = self.link_importance(encoded_spans)
                # attn_bias = weights[:,0].unsqueeze(-1).unsqueeze(-1)*link_attn_bias + weights[:,1].unsqueeze(-1).unsqueeze(-1)*edges_value_graph
                # attn_bias = link_attn_bias + torch.softmax(edges_value_graph,dim=-1)
                attn_bias = link_attn_bias
                # attn_bias = None
                # attention_mask = None
                node = enc_layer(node, attn_bias, attention_mask)
            node = self.final_ln(node)

            # node_info = node[:,1:,:]

            # node, node_weight = self._gcn(node=node, node_mask=span_mask,
            #                               argument_graph=argument_graph,
            #                               punctuation_graph=punctuation_graph)

            gcn_info_vec = self.get_gcn_info_vector(node_in_seq_indices, node,
                                                    size=sequence_output.size(), device=sequence_output.device)

            gcn_updated_sequence_output = self._gcn_enc(
                self._gcn_prj_ln(sequence_output + gcn_info_vec))  # [batchsize*n_choice, seq_len, hidden_size]

            # passage hidden and question hidden
            sequence_h2_weight = self._proj_sequence_h(gcn_updated_sequence_output).squeeze(-1)
            passage_h2_weight = util.masked_softmax(sequence_h2_weight.float(), flat_passage_mask.float())
            passage_h2 = util.weighted_sum(gcn_updated_sequence_output, passage_h2_weight)
            question_h2_weight = util.masked_softmax(sequence_h2_weight.float(), flat_question_mask.float())
            question_h2 = util.weighted_sum(gcn_updated_sequence_output, question_h2_weight)

            gcn_output_feats = torch.cat([passage_h2, question_h2, gcn_updated_sequence_output[:, 0]], dim=1)
            gcn_logits = self._proj_span_num(gcn_output_feats)

        if self.use_pool:
            ''' The baseline branch. The output. '''
            pooled_output = self.dropout(pooled_output)
            baseline_logits = self.classifier(pooled_output)

        if self.use_gcn and self.use_pool:
            ''' Merge gcn_logits & baseline_logits. TODO: different way of merging. '''

            if self.merge_type == 1:
                logits = gcn_logits + baseline_logits

            elif self.merge_type == 2:
                pooled_output = self.dropout(pooled_output)
                merged_feats = torch.cat([gcn_updated_sequence_output[:, 0], pooled_output], dim=1)
                logits = self._proj_gcn_pool_3(merged_feats)

            elif self.merge_type == 3:
                pooled_output = self.dropout(pooled_output)
                merged_feats = torch.cat([gcn_updated_sequence_output[:, 0], pooled_output,
                                          gcn_updated_sequence_output[:, 0], pooled_output], dim=1)
                logits = self._proj_gcn_pool_4(merged_feats)

            elif self.merge_type == 4:
                pooled_output = self.dropout(pooled_output)
                merged_feats = torch.cat([passage_h2, question_h2, pooled_output], dim=1)
                logits = self._proj_gcn_pool(merged_feats)

            elif self.merge_type == 5:
                pooled_output = self.dropout(pooled_output)
                merged_feats = torch.cat([passage_h2, question_h2, gcn_updated_sequence_output[:, 0], pooled_output],
                                         dim=1)
                logits = self._proj_gcn_pool_4(merged_feats)

        elif self.use_gcn:
            logits = gcn_logits
        elif self.use_pool:
            logits = baseline_logits
        else:
            raise Exception

        reshaped_logits = logits.squeeze(-1).view(-1, num_choices)
        outputs = (reshaped_logits,)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)

            outputs = (loss,) + outputs
        return outputs