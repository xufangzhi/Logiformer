'''
Adapted from https://github.com/llamazing/numnet_plus
'''

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from tools import allennlp as util


def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def swish(x):
    return x * torch.sigmoid(x)


class ResidualGRU(nn.Module):
    def __init__(self, hidden_size, dropout=0, num_layers=2):
        super(ResidualGRU, self).__init__()
        self.enc_layer = nn.GRU(input_size=hidden_size, hidden_size=hidden_size // 2, num_layers=num_layers,
                                batch_first=True, dropout=dropout, bidirectional=True)
        self.enc_ln = nn.LayerNorm(hidden_size)

    def forward(self, input):
        output, _ = self.enc_layer(input)
        return self.enc_ln(output + input)



class FFNLayer(nn.Module):
    def __init__(self, input_dim, intermediate_dim, output_dim, dropout, layer_norm=True):
        super(FFNLayer, self).__init__()
        self.fc1 = nn.Linear(input_dim, intermediate_dim)
        if layer_norm:
            self.ln = nn.LayerNorm(intermediate_dim)
        else:
            self.ln = None
        self.dropout_func = nn.Dropout(dropout)
        self.fc2 = nn.Linear(intermediate_dim, output_dim)

    def forward(self, input):
        inter = self.fc1(self.dropout_func(input))
        inter_act = gelu(inter)
        if self.ln:
            inter_act = self.ln(inter_act)
        return self.fc2(inter_act)


class ArgumentGCN(nn.Module):

    def __init__(self, node_dim, extra_factor_dim=0, iteration_steps=1):
        super(ArgumentGCN, self).__init__()

        self.node_dim = node_dim
        self.iteration_steps = iteration_steps

        self._node_weight_fc = torch.nn.Linear(node_dim + extra_factor_dim, 1, bias=True)

        self._self_node_fc = torch.nn.Linear(node_dim, node_dim, bias=True)

        self._node_fc_argument = torch.nn.Linear(node_dim, node_dim,
                                                 bias=False)  
        self._node_fc_punctuation = torch.nn.Linear(node_dim, node_dim,
                                                    bias=False)  

    def forward(self,
                node,  
                node_mask,  
                argument_graph,  
                punctuation_graph,  
                extra_factor=None):
        ''' '''
        '''
        Current: 2 relation patterns.
            - argument edge. (most of them are causal relations)
            - punctuation edges. (including periods and commas)
        '''
        node_len = node.size(1)

        diagmat = torch.diagflat(torch.ones(node.size(1), dtype=torch.long,
                                            device=node.device))  
        diagmat = diagmat.unsqueeze(0).expand(node.size(0), -1, -1)  
        dd_graph = node_mask.unsqueeze(1) * node_mask.unsqueeze(-1) * (1 - diagmat)  

        graph_argument = dd_graph * argument_graph
        graph_punctuation = dd_graph * punctuation_graph

        node_neighbor_num = graph_argument.sum(-1) + graph_punctuation.sum(-1)
        node_neighbor_num_mask = (node_neighbor_num >= 1).long()
        node_neighbor_num = util.replace_masked_values(node_neighbor_num.float(), node_neighbor_num_mask, 1)  # replace 0 with 1

        all_weight = []
        for step in range(self.iteration_steps):

            ''' (1) Node Relatedness Measure '''
            if extra_factor is None:
                d_node_weight = torch.sigmoid(self._node_weight_fc(node)).squeeze(
                    -1)

            else:
                d_node_weight = torch.sigmoid(self._node_weight_fc(torch.cat((node, extra_factor), dim=-1))).squeeze(
                    -1)  

            all_weight.append(d_node_weight)  

            self_node_info = self._self_node_fc(node) 

            ''' (2) Message Propagation (each relation type) '''
            
            node_info_argument = self._node_fc_argument(node)
            node_weight = util.replace_masked_values(                      # update the node which has connection
                d_node_weight.unsqueeze(1).expand(-1, node_len, -1),
                graph_argument,
                0)  
            node_info_argument = torch.matmul(node_weight, node_info_argument)  

            
            node_info_punctuation = self._node_fc_punctuation(node)
            node_weight = util.replace_masked_values(
                d_node_weight.unsqueeze(1).expand(-1, node_len, -1),
                graph_punctuation,
                0)  
            node_info_punctuation = torch.matmul(node_weight, node_info_punctuation)

            agg_node_info = (node_info_argument + node_info_punctuation) / node_neighbor_num.unsqueeze(-1)

            ''' (3) Node Representation Update '''
            node = F.relu(self_node_info + agg_node_info)

        all_weight = [weight.unsqueeze(1) for weight in all_weight]
        all_weight = torch.cat(all_weight, dim=1)

        return node, all_weight


class ArgumentGCN_wreverseedges_double(nn.Module):

    def __init__(self, node_dim, extra_factor_dim=0, iteration_steps=1):
        super(ArgumentGCN_wreverseedges_double, self).__init__()

        self.node_dim = node_dim
        self.iteration_steps = iteration_steps

        self._node_weight_fc = torch.nn.Linear(node_dim + extra_factor_dim, 1, bias=True)

        self._self_node_fc = torch.nn.Linear(node_dim, node_dim, bias=True)

        self._node_fc_argument = torch.nn.Linear(node_dim, node_dim,
                                                 bias=False)  
        self._node_fc_punctuation = torch.nn.Linear(node_dim, node_dim,
                                                    bias=False)  
        self._node_fc_argument_prime = torch.nn.Linear(node_dim, node_dim,
                                                       bias=False)  
        self._node_fc_punctuation_prime = torch.nn.Linear(node_dim, node_dim,
                                                          bias=False)  

        self._node_fc_argument_2 = torch.nn.Linear(node_dim, node_dim,
                                                   bias=False)  
        self._node_fc_punctuation_2 = torch.nn.Linear(node_dim, node_dim,
                                                      bias=False)  
        self._node_fc_argument_prime_2 = torch.nn.Linear(node_dim, node_dim,
                                                         bias=False)  
        self._node_fc_punctuation_prime_2 = torch.nn.Linear(node_dim, node_dim,
                                                            bias=False)  

    def forward(self,
                node,  
                node_mask,  
                argument_graph,  
                punctuation_graph,  
                extra_factor=None):
        ''' '''
        '''
        Current: 2 relation patterns & reversed directed edges.
            - argument edge. (most of them are causal relations)
            - punctuation edges. (including periods and commas)

        Date: 20/11/2020
        Change log: 
            * add double node_info.
        '''

        # node_mask = [batchsize * n_choices, node_len], 0 is padding
        # node = [batchsize * n_choices, node_len, embed_size]
        node_len = node.size(1)

        diagmat = torch.diagflat(torch.ones(node.size(1), dtype=torch.long,       # matric with 1 in the diag
                                            device=node.device))  
        diagmat = diagmat.unsqueeze(0).expand(node.size(0), -1, -1)     # expand the diagmat to more (the num is equal to bz*n_choice)   [bz*n_choice, node_len, node_len]
        dd_graph = node_mask.unsqueeze(1) * node_mask.unsqueeze(-1) * (1 - diagmat)  # set diag to 0, and other actual nodes to 1

        graph_argument = dd_graph * argument_graph
        graph_punctuation = dd_graph * punctuation_graph

        ''' The reverse directed edges. '''
        graph_argument_re = dd_graph * graph_argument.permute(0, 2, 1)
        graph_punctuation_re = dd_graph * graph_punctuation.permute(0, 2, 1)

        node_neighbor_num = graph_argument.sum(-1) + graph_punctuation.sum(-1) + \
                            graph_argument_re.sum(-1) + graph_punctuation_re.sum(-1)
        node_neighbor_num_mask = (node_neighbor_num >= 1).long()
        node_neighbor_num = util.replace_masked_values(node_neighbor_num.float(), node_neighbor_num_mask, 1)

        all_weight = []
        for step in range(self.iteration_steps):

            ''' (1) Node Relatedness Measure '''
            if extra_factor is None:
                d_node_weight = torch.sigmoid(self._node_weight_fc(node)).squeeze(
                    -1)  
            else:
                d_node_weight = torch.sigmoid(self._node_weight_fc(torch.cat((node, extra_factor), dim=-1))).squeeze(
                    -1)  

            all_weight.append(d_node_weight)  

            self_node_info = self._self_node_fc(node)  

            ''' (2) Message Propagation (each relation type) '''
            
            node_info_argument = self._node_fc_argument(node)
            node_info_argument_2 = self._node_fc_argument_2(node)
            node_weight = util.replace_masked_values(
                d_node_weight.unsqueeze(1).expand(-1, node_len, -1),
                graph_argument,
                0)  
            node_info_argument = torch.matmul(node_weight, node_info_argument)
            node_info_argument_2 = torch.matmul(node_weight, node_info_argument_2)

            
            node_info_argument_prime = self._node_fc_argument_prime(node)
            node_info_argument_prime_2 = self._node_fc_argument_prime_2(node)
            node_weight = util.replace_masked_values(
                d_node_weight.unsqueeze(1).expand(-1, node_len, -1),
                graph_argument_re,
                0)  
            node_info_argument_prime = torch.matmul(node_weight, node_info_argument_prime)
            node_info_argument_prime_2 = torch.matmul(node_weight, node_info_argument_prime_2)

            
            node_info_punctuation = self._node_fc_punctuation(node)
            node_info_punctuation_2 = self._node_fc_punctuation_2(node)
            node_weight = util.replace_masked_values(
                d_node_weight.unsqueeze(1).expand(-1, node_len, -1),
                graph_punctuation,
                0)  
            node_info_punctuation = torch.matmul(node_weight, node_info_punctuation)
            node_info_punctuation_2 = torch.matmul(node_weight, node_info_punctuation_2)

            
            node_info_punctuation_prime = self._node_fc_punctuation_prime(node)
            node_info_punctuation_prime_2 = self._node_fc_punctuation_prime_2(node)
            node_weight = util.replace_masked_values(
                d_node_weight.unsqueeze(1).expand(-1, node_len, -1),
                graph_punctuation_re,
                0)  
            node_info_punctuation_prime = torch.matmul(node_weight, node_info_punctuation_prime)
            node_info_punctuation_prime_2 = torch.matmul(node_weight, node_info_punctuation_prime_2)

            agg_node_info = (node_info_argument + node_info_punctuation +
                             node_info_argument_prime + node_info_punctuation_prime +
                             node_info_argument_2 + node_info_punctuation_2 +
                             node_info_argument_prime_2 + node_info_punctuation_prime_2
                             ) / node_neighbor_num.unsqueeze(-1)

            ''' (3) Node Representation Update '''
            node = F.relu(self_node_info + agg_node_info)

        all_weight = [weight.unsqueeze(1) for weight in all_weight]
        all_weight = torch.cat(all_weight, dim=1)

        return node, all_weight











