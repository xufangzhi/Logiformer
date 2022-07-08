# -*- coding: UTF-8 -*-
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import numpy as np
from typing import List, Dict, Any, Tuple
from itertools import groupby
from operator import itemgetter
import copy
from util import FFNLayer, ResidualGRU
from tools import allennlp as util
from transformers import BertPreTrainedModel, RobertaModel, BertModel, AlbertModel, XLNetModel, RobertaForMaskedLM, RobertaTokenizer
import torch.nn.functional as F
import math


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

class qtype_Embedding(nn.Module):
    def __init__(self, hidden_size):
        super(qtype_Embedding, self).__init__()
        self.hidden_size = hidden_size

    def forward(self, x):  # input is encoded spans
        batch_size = x.size(0)
        type_num = 19
        qtype_embed = torch.arange(type_num, dtype=torch.long)
        self.embedding = nn.Embedding(type_num, self.hidden_size)  # position embedding
        qtype_embed = self.embedding(qtype_embed)
        for i in range(batch_size):
            if i == 0:
                final_embed = qtype_embed[x[i].item(), :].unsqueeze(0)
            else:
                final_embed = torch.cat((final_embed, qtype_embed[x[i].item(), :].unsqueeze(0)), 0)

        return final_embed.to("cuda")


# normal position embedding
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



class Logiformer(BertPreTrainedModel):

    def __init__(self,
                 config,
                 init_weights: bool,
                 max_rel_id,
                 hidden_size: int,
                 dropout_prob: float = 0.1,
                 token_encoder_type: str = "roberta"):
        super().__init__(config)

        self.layer_num = 5
        self.head_num = 5
        self.token_encoder_type = token_encoder_type
        self.max_rel_id = max_rel_id

        ''' roberta model '''
        self.roberta = RobertaModel(config)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)

        self._gt_prj_ln = nn.LayerNorm(hidden_size)
        self._gt_enc = nn.Sequential(nn.Linear(hidden_size, hidden_size, bias=False), nn.ReLU())

        self._proj_sequence_h = nn.Linear(hidden_size, 1, bias=False)

        self._proj_span_num = FFNLayer(3 * hidden_size, hidden_size, 1, dropout_prob)
        self._proj_gt_pool = FFNLayer(3 * hidden_size, hidden_size, 1, dropout_prob)

        self.pre_ln = nn.LayerNorm(hidden_size)

        self.pos_embed = Position_Embedding(hidden_size)

        self.input_dropout = nn.Dropout(dropout_prob)
        encoders = [EncoderLayer(hidden_size, hidden_size, dropout_prob, dropout_prob, self.head_num)
                    for _ in range(self.layer_num)]
        self.encoder_layers = nn.ModuleList(encoders)

        self.final_ln = nn.LayerNorm(hidden_size)

        if init_weights:
            self.init_weights()

    def split_into_spans_9(self, seq, seq_mask, split_bpe_ids, passage_mask, option_mask, question_mask, type):
        '''
            this function is modified from DAGN
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
            if type == "causal":
                split_ids_indices = np.where(item_split_ids > 0)[0].tolist()     # causal type
            else:
                split_ids_indices = np.where(item_split_ids > 0)[0].tolist()     # Co-reference type

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
                    # span = item_seq[grouped_split_ids_indices[i][-1] + 1:grouped_split_ids_indices[i + 1][0]]
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

    def get_gt_info_vector(self, indices, node, size, device):
        '''
        give the node embed to each token in one node

        :param indices: list(len=bsz) of list(len=n_notes) of list(len=varied).
        :param node: (bsz, n_nodes, embed_size)
        :param size: value=(bsz, seq_len, embed_size)
        :param device:
        :return:
        '''
        batch_size = size[0]
        gt_info_vec = torch.zeros(size=size, dtype=torch.float, device=device)

        for b in range(batch_size):
            for ids, emb in zip(indices[b], node[b]):
                gt_info_vec[b, ids] = emb
            gt_info_vec[b, 0] = node[b].mean(0)   # global feature
        return gt_info_vec

    def get_adjacency_matrices_2(self, edges: List[List[int]], coocc_tags,
                                 n_nodes: int, device: torch.device, type: str):
        '''
        this function is modified from DAGN
        Convert the edge_value_list into adjacency matrices.
            * argument graph adjacency matrix. Asymmetric (directed graph).
            * punctuation graph adjacency matrix. Symmetric (undirected graph).

            : argument
                - edges:list[list[str]]. len_out=(bsz x n_choices), len_in=n_edges. value={-1, 0, 1, 2, 3, 4, 5}.

        '''
        batch_size = len(edges)
        hidden_size = 1024
        argument_graph = torch.zeros(
            (batch_size, n_nodes, n_nodes))  # NOTE: the diagonal should be assigned 0 since is acyclic graph.
        punct_graph = torch.zeros(
            (batch_size, n_nodes, n_nodes))  # NOTE: the diagonal should be assigned 0 since is acyclic graph.
        causal_graph = torch.zeros(
            (batch_size, n_nodes, n_nodes))  # NOTE: the diagonal should be assigned 0 since is acyclic graph.
        for b, sample_edges in enumerate(edges):
            for i, edge_value in enumerate(sample_edges):
                if edge_value == 1:  
                    try:
                        argument_graph[b, i + 1, i + 2] = 1
                    except Exception:
                        pass
                elif edge_value == 2:  
                    argument_graph[b, i, i + 1] = 1
                elif edge_value == 3: 
                    argument_graph[b, i + 1, i] = 1
                    causal_graph[b, i, i + 1] = 1
                    # causal_graph[b, i + 1, i] = 1
                elif edge_value == 4: 
                    argument_graph[b, i, i + 1] = 1
                    # argument_graph[b, i + 1, i] = 1
                elif edge_value == 5: 
                    try:
                        punct_graph[b, i, i + 1] = 1
                        punct_graph[b, i + 1, i] = 1
                    except Exception:
                        pass
        ''' coocc tag calculate '''
        coocc_graph = torch.zeros(
            (batch_size, n_nodes, n_nodes), dtype=torch.float)  # NOTE: the diagonal should be assigned 0 since is acyclic graph.
        if type == "coocc":
            for b, sample_coocc in enumerate(coocc_tags):
                for i, tag in enumerate(sample_coocc):
                    if tag[0].item() != -1:
                        coocc_graph[b, int(tag[0].item()), int(tag[1].item())] = 1
                        coocc_graph[b, int(tag[1].item()), int(tag[0].item())] = 1
            # for i in range(coocc_graph.size(1)):
            #     coocc_graph[:, i, i] = 1
        return argument_graph.to(device), punct_graph.to(device), causal_graph.to(device), coocc_graph.to(device)


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
                coocc: torch.LongTensor,
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
        flat_coocc_tags = coocc.view(-1, coocc.size(-2), coocc.size(-1)) if coocc is not None else None
        # flat_qtype = qtype.view(-1) if qtype is not None else None
        bert_outputs = self.roberta(flat_input_ids, attention_mask=flat_attention_mask, token_type_ids=None)
        sequence_output = bert_outputs[0]
        pooled_output = bert_outputs[1]  # [bz*n_choice, hidden_size]

        # Logiformer Block
        new_punct_id = self.max_rel_id + 1   # new_punct_id:5
        new_punct_bpe_ids = new_punct_id * flat_punct_bpe_ids  # punct_id: 1 -> 5. for incorporating with argument_bpe_ids.
        _flat_all_bpe_ids = flat_argument_bpe_ids + new_punct_bpe_ids  # -1:padding, 0:non, 1-4: arg, 5:punct.
        overlapped_punct_argument_mask = (_flat_all_bpe_ids > new_punct_id).long()
        flat_all_bpe_ids = _flat_all_bpe_ids * (
                    1 - overlapped_punct_argument_mask) + flat_argument_bpe_ids * overlapped_punct_argument_mask
        assert flat_argument_bpe_ids.max().item() <= new_punct_id

        # encoded_spans: (bsz x n_choices, n_nodes, embed_size)
        # span_mask: (bsz x n_choices, n_nodes)
        # edges: list[list[int]]
        # node_in_seq_indices: list[list[list[int]]]

        ''' Logical Causal '''

        encoded_spans, span_mask, edges, edges_embed, node_in_seq_indices, attention_mask, ques_seq = self.split_into_spans_9(
            sequence_output,
            flat_attention_mask,
            flat_all_bpe_ids,
            flat_passage_mask,
            flat_option_mask,
            flat_question_mask,
            "causal")
        argument_graph, punctuation_graph, causal_graph, coocc_graph = self.get_adjacency_matrices_2(
            edges, coocc_tags=flat_coocc_tags, n_nodes=encoded_spans.size(1), device=encoded_spans.device, type="causal")
        encoded_spans = encoded_spans + self.pos_embed(encoded_spans)  # node_embedding + positional embedding

        node = self.input_dropout(encoded_spans)
        causal_layer_output_list = []
        for enc_layer in self.encoder_layers:
            attn_bias = causal_graph
            node = enc_layer(node, attn_bias, attention_mask)
            causal_layer_output_list.append(node)
        node_causal = causal_layer_output_list[-1] + causal_layer_output_list[-2]
        node_causal = self.final_ln(node_causal)
        gt_info_vec_causal = self.get_gt_info_vector(node_in_seq_indices, node_causal,
                                                size=sequence_output.size(), device=sequence_output.device)


        ''' Co-occurrence Semantic '''

        encoded_spans, span_mask, edges, edges_embed, node_in_seq_indices, attention_mask, ques_seq1 = self.split_into_spans_9(
            sequence_output,
            flat_attention_mask,
            flat_all_bpe_ids,
            flat_passage_mask,
            flat_option_mask,
            flat_question_mask,
            "coocc")
        argument_graph, punctuation_graph, causal_graph, coocc_graph = self.get_adjacency_matrices_2(
            edges, coocc_tags=flat_coocc_tags, n_nodes=encoded_spans.size(1), device=encoded_spans.device, type="coocc")
        encoded_spans = encoded_spans + self.pos_embed(encoded_spans)  # node_embedding + positional embedding
        node = self.input_dropout(encoded_spans)
        coocc_layer_output_list = []
        for enc_layer in self.encoder_layers:
            attn_bias = coocc_graph
            node = enc_layer(node, attn_bias, attention_mask)
            coocc_layer_output_list.append(node)
        node_coocc = coocc_layer_output_list[-1] + coocc_layer_output_list[-2]
        node_coocc = self.final_ln(node_coocc)
        gt_info_vec_coocc = self.get_gt_info_vector(node_in_seq_indices, node_coocc,
                                                size=sequence_output.size(), device=sequence_output.device)   # [batchsize*n_choice, seq_len, hidden_size]

        gt_updated_sequence_output = self._gt_enc(
            self._gt_prj_ln(sequence_output + 0.6*gt_info_vec_coocc + 0.4*gt_info_vec_causal))

        # passage hidden and question hidden
        sequence_h2_weight = self._proj_sequence_h(gt_updated_sequence_output).squeeze(-1)
        passage_h2_weight = util.masked_softmax(sequence_h2_weight.float(), flat_passage_mask.float())
        passage_h2 = util.weighted_sum(gt_updated_sequence_output, passage_h2_weight)
        question_h2_weight = util.masked_softmax(sequence_h2_weight.float(), flat_question_mask.float())
        question_h2 = util.weighted_sum(gt_updated_sequence_output, question_h2_weight)

        ''' obtain logits '''
        gt_output_feats = torch.cat([passage_h2, question_h2, gt_updated_sequence_output[:, 0]], dim=1)
        gt_logits = self._proj_span_num(gt_output_feats)


        pooled_output = self.dropout(pooled_output)
        merged_feats = torch.cat([passage_h2, question_h2, pooled_output], dim=1)
        logits = self._proj_gt_pool(merged_feats)
        logits = logits + 0.5*gt_logits


        reshaped_logits = logits.squeeze(-1).view(-1, num_choices)
        outputs = (reshaped_logits,)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)

            outputs = (loss,) + outputs
        return outputs