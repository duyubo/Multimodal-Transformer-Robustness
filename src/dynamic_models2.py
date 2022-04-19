from re import I
from typing import ChainMap
import torch
from torch import nn
import torch.nn.functional as F
import sys
"""!!!!!!!!!!!!!!!!!!!!!!! To be deleted !!!!!!!!!!!!!!!!!!!!!!!"""
sys.path.append('/content/drive/MyDrive/Colab_Notebooks/Multimodal-Transformer-Robustness/')
"""!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"""

from modules.transformer import TransformerEncoder
from modules.dynamic_transformer import DynamicTransformerEncoder
from src.models2 import  *
from modules.dynamic_layers import DynamicLinear, DynamicLayerNorm


class DynamicMULTModel(MULTModel):
    """ 
                        structure of the modified MULT model

                                      output
                                        ^
                                        |
                                      linear
                                        ^
                                        |
                                      linear
                                        ^
                                        |
          --------------------------------------------------------------------
          |                                                                   |
    self1:       all of below              all of below                all of below
                           ^ ^ ^                     ^ ^ ^                      ^ ^ ^
                           | | |                     | | |                      | | |
    cross:  AV->L, AL->V --| | |       VA->L, VL->A--| | |        LA->V, LV->A--| | |
    cross:  A -> V, A -> L --| |       V -> A, V -> L--| |        L -> A, L -> V--| |
    self0:  A -> A ------------|       V -> V------------|        L -> L------------|
    conv    A                          V                          L
    """
    def __init__(self, origin_dimensions:list, dimension, 
        num_heads, head_dim, layers_hybrid_attn, layers_self_attn, attn_dropout:list, 
        relu_dropout, res_dropout, out_dropout, embed_dropout, attn_mask, output_dim, modality_set):
        
        nn.Module.__init__(self)
        """ Fixed Hyperparameters """
        self.orig_dimensions = origin_dimensions
        self.d = dimension
        self.attn_dropout = attn_dropout
        assert len(self.attn_dropout) == len(self.orig_dimensions) + 1
        self.relu_dropout = relu_dropout
        self.res_dropout = res_dropout
        self.out_dropout = out_dropout
        self.embed_dropout = embed_dropout
        self.attn_mask = attn_mask
        self.output_dim = output_dim # should be same as the label dimension
        self.modality_list = modality_set
        self.m = ModalityStr(modality_set)

        """ Shrinkable Hyperparameters
        in the parent graph we set up each modality with the same number of layers, hidden dimensions, head numbers and etc. 
        But the number of layers, head numbers, head dim for each modality are not required to be same during sampling!
        Output dimension of the temporal conv layers should always be the same """
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.layers_hybrid_attn = layers_hybrid_attn
        self.layers_self_attn = layers_self_attn
        self.modality_num = len(self.orig_dimensions)
        self.combined_dim = AmnSum(self.modality_num) * self.d
        
        """ Temporal Convolutional Layers (None Shrinkable) """
        self.proj = [nn.Conv1d(self.orig_dimensions[i], self.d, kernel_size=3, padding=0, bias=False) for i in range(self.modality_num)]
        self.proj = nn.ModuleList(self.proj)

        """ Self Attentions (Shrinkable) self0 """
        self.trans_mems0 = nn.ModuleDict({'mems0' + self.modality_list[i]: self.get_network(i, 0, mem=False, layers = self.layers_hybrid_attn) for i in range(self.modality_num)})
        print('mems0: ', self.trans_mems0.keys())

        """ cross Attentions (Shrinkable) cross """
        modality_combines = self.m.gen_modality_str_all()
        self.trans = nn.ModuleDict({'cross' + modality_combines[i]: self.get_network(i, i, mem=False, layers=self.layers_hybrid_attn) for i in range(len(modality_combines))})
        print('trans: ', self.trans.keys())
        
        self.modality_index_list = []
        for i in self.modality_list:
            modality_str = [i] + self.m.gen_modality_str_all(modality_set = [i])
            modality_str_with_index = {modality_str[i]: i for i in range(len(modality_str))}
            self.modality_index_list.append(modality_str_with_index)
        print('modality index list: ', self.modality_index_list)

        """ Self Attentions (Shrinkable) self1 """
        self.trans_mems =nn.ModuleDict({'mems' + self.modality_list[i]: self.get_network(i, i, mem=True, layers=self.layers_self_attn) for i in range(self.modality_num)})
        print('mems: ', self.trans_mems.keys())

        """ Projection Layers (Shrinkable) """
        self.proj1 = DynamicLinear(self.combined_dim, self.combined_dim, bias = True)
        self.proj2 = DynamicLinear(self.combined_dim, self.combined_dim, bias = True)
        self.out_layer = DynamicLinear(self.combined_dim, self.output_dim, bias = True)

        #default active_modality is full
        self.active_modality = list(range(self.modality_num))
        # default active cross attention modules are the same as MULT-Transformer
        # must be in the same order as modality_combines
        self.active_cross = [self.m.gen_modality_str(i) for i in self.modality_list]
        self.active_cross_output = [self.m.gen_modality_str(i) for i in self.modality_list]

        super(DynamicMULTModel, self).__init__(
            proj = self.proj, trans_mems0 = self.trans_mems0, trans = self.trans, trans_mems = self.trans_mems, 
            proj1 = self.proj1, proj2 = self.proj2, out_layer = self.out_layer,
            origin_dimensions = self.orig_dimensions, dimension = self.d, 
            num_heads = self.num_heads, head_dim = self.head_dim, layers_hybrid_attn = self.layers_hybrid_attn,
            layers_self_attn = self.layers_self_attn, attn_dropout = self.attn_dropout, 
            relu_dropout = self.relu_dropout, res_dropout = self.res_dropout, out_dropout = self.out_dropout, 
            embed_dropout = self.embed_dropout, attn_mask = self.attn_mask, output_dim = self.output_dim,
            cross = self.active_cross.copy(), cross_output = self.active_cross_output.copy(), 
            modality_list = self.modality_list.copy())

    def get_network(self, mod1, mod2, mem, layers=-1):
        if not mem:
            embed_dim_in = self.d
            if mod2 == 0:
              attn_dropout = self.attn_dropout[mod1]
            else:
              attn_dropout = 0
        else:
            embed_dim_in = int(self.combined_dim/self.modality_num)
            attn_dropout = self.attn_dropout[-1]

        return DynamicTransformerEncoder(embed_dim = embed_dim_in,
                                  head_dim = self.head_dim,
                                  num_heads = self.num_heads,
                                  layers = layers,
                                  attn_dropout=attn_dropout,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout,
                                  attn_mask=self.attn_mask)
    
    def forward(self, x): 
        assert len(x) == self.modality_num # missing modality will be repalced by ones or zeros, can not be deleted
        x = [v.permute(0, 2, 1) for v in x]  # n_modalities * [batch_size, n_features, seq_len]
        proj_x = [self.proj[i](x[i]) for i in self.active_modality]
        proj_x = torch.stack(proj_x)
        proj_x = proj_x.permute(0, 3, 1, 2)

        """ self attention of each modality"""
        proj_x1 = {self.modality_list[self.active_modality[i]]: self.trans_mems0['mems0' + self.modality_list[self.active_modality[i]]](proj_x[i]) for i in range(len(self.active_modality))}
        h_ = proj_x1

        """ multi level cross attention """
        last_hs = []
        active_mask_output = []
        for i in self.active_modality:
            for modality_cross in self.active_cross[i]:
                h_[modality_cross] = self.trans['cross' + modality_cross](h_[modality_cross[-1]], h_[modality_cross[:-1]], h_[modality_cross[:-1]])
          
            h = torch.cat([h_[modality_cross] for modality_cross in self.active_cross_output[i]], dim = 2)
            active_mask = []
            for m_c in self.active_cross_output[i]:
                index_cross = self.modality_index_list[i][m_c]
                active_mask.extend(list(range(index_cross * self.d, (index_cross + 1) * self.d )))
                active_mask_output.extend(list(range(
                                      self.d * len(self.modality_index_list[i]) * i + index_cross * self.d, 
                                      self.d * len(self.modality_index_list[i]) * i + (index_cross + 1) * self.d 
                                      )))
            active_mask = torch.tensor(active_mask).type(torch.IntTensor).to(next(self.parameters()).device)
            """self attention in the highest level"""
            h = self.trans_mems['mems' + self.modality_list[i]](h, active_mask = active_mask)
            last_hs.append(h[-1])
        
        out = torch.cat(last_hs, dim=1)
        active_indexes = torch.Tensor(active_mask_output).type(torch.IntTensor).to(next(self.parameters()).device)

        """ Concatenation layer"""   
        out_proj = self.proj2(
            F.dropout(
              F.relu(
                self.proj1(out, active_dim_in = None, active_dim_out = None, mask_in = active_indexes, mask_out = [None])
              ), 
              p = self.out_dropout, 
              training = self.training
            ), active_dim_in = None, active_dim_out = None, mask_in = [None], mask_out = active_indexes
        )
        out_proj += out
        out = self.out_layer(out_proj, active_dim_in = None, active_dim_out = None, mask_in = active_indexes, mask_out = [None])
        return out

    def get_active_subnet(self, active_self_attn_layer_num, 
                          active_hybrid_attn_layer_num, 
                          active_dimension, 
                          active_head_num, 
                          active_head_dim, 
                          active_modality:list, # list of int
                          active_cross:list, # 2-d list of string
                          active_cross_output:list):# 2-d list of string
        #print(active_self_attn_layer_num, active_hybrid_attn_layer_num, active_dimension, active_head_num , active_head_dim , active_modality)
        #print('active modality in get_active_subnet', active_modality, active_cross, active_cross_output)
        """copy projecction (conv) layers"""
        proj = []
        for i in active_modality:
            p = nn.Conv1d(self.orig_dimensions[i], self.d, kernel_size=3, padding=0, bias=False)
            p.weight.data.copy_(self.proj[i].weight.data)
            proj.append(p)
        proj = nn.ModuleList(proj)

        """copyy first self attention modules"""
        trans_mems0 = {'mems0'+self.modality_list[i]: self.trans_mems0['mems0'+self.modality_list[i]].get_active_subnet(
                                    active_layer_num = active_hybrid_attn_layer_num, 
                                    active_dimension = active_dimension, 
                                    active_head_num = active_head_num, 
                                    active_head_dim = active_head_dim, 
                                    active_mask = [None]
                                    ) for i in active_modality}
        
        trans_mems0 = nn.ModuleDict(trans_mems0)

        """copy cross attention modules"""
        trans = {}
        for i in active_modality:
            for cross_modality in active_cross[i]:
              trans['cross' + cross_modality] = self.trans['cross' + cross_modality].get_active_subnet(
                                    active_layer_num = active_hybrid_attn_layer_num, 
                                    active_dimension = active_dimension, 
                                    active_head_num = active_head_num, 
                                    active_head_dim = active_head_dim, 
                                    active_mask = [None]
                                    )
        trans = nn.ModuleDict(trans)

        """copy the highest level self attention module"""
        trans_mems = {}
        active_mask_output = []
        for i in active_modality:
            active_mask = []
            for m_c in active_cross_output[i]:
                index_cross = self.modality_index_list[i][m_c]
                active_mask.extend(list(range(index_cross * self.d, (index_cross + 1) * self.d )))
                active_mask_output.extend(list(range(
                                      self.d * len(self.modality_index_list[i]) * i + index_cross * self.d, 
                                      self.d * len(self.modality_index_list[i]) * i + (index_cross + 1) * self.d 
                                      )))
            active_mask = torch.tensor(active_mask).type(torch.IntTensor).to(next(self.parameters()).device)   
            #print('masks for self atten layer: ', i, active_mask)                       
            trans_mems['mems' + self.modality_list[i]] = self.trans_mems['mems' + self.modality_list[i]].get_active_subnet(
                                      active_layer_num = active_self_attn_layer_num, 
                                      active_dimension = active_dimension, 
                                      active_head_num = active_head_num, 
                                      active_head_dim = active_head_dim, 
                                      active_mask = active_mask)
        trans_mems = nn.ModuleDict(trans_mems)

        """copy concatanation layer"""
        active_indexes = torch.Tensor(active_mask_output).type(torch.IntTensor).to(next(self.parameters()).device)
        #print('masks for concatation layer: ', active_indexes)
        proj1 = self.proj1.copy(dim_in = None, dim_out = None, mask_in = active_indexes, mask_out = [None])
        proj2 = self.proj2.copy(dim_in = None, dim_out = None, mask_in = [None], mask_out = active_indexes)
        out_layer = self.out_layer.copy(dim_in = None, dim_out = None, mask_in = active_indexes, mask_out = [None])

        attn_drop = [self.attn_dropout[i] for i in active_modality]
        attn_drop.append(self.attn_dropout[-1])

        cross = []
        cross_output = []
        for i in range(len(active_cross_output)):
          if active_cross_output[i]:
            cross.append(active_cross[i])
            cross_output.append(active_cross_output[i])
        assert len(cross_output) == len(active_modality)
        
        print('cross', cross, 'cross_output', cross_output)
        model = MULTModel(
            proj = proj, trans_mems0 = trans_mems0, trans = trans, trans_mems = trans_mems, 
            proj1 = proj1, proj2 = proj2, out_layer = out_layer,
            origin_dimensions = [self.orig_dimensions[i] for i in active_modality], dimension = self.d, 
            num_heads = active_head_num, head_dim = active_head_dim, layers_hybrid_attn = active_hybrid_attn_layer_num,
            layers_self_attn = active_self_attn_layer_num, attn_dropout = attn_drop, 
            relu_dropout = self.relu_dropout, res_dropout = self.res_dropout, out_dropout = self.out_dropout, 
            embed_dropout = self.embed_dropout, attn_mask = self.attn_mask, output_dim = self.output_dim,
            cross = cross, cross_output = cross_output, modality_list = [self.modality_list[i] for i in active_modality])
        
        model = model.to(self.parameters().__next__().device)
        return model

    def set_active(self, active_self_attn_layer_num, active_hybrid_attn_layer_num, 
                  active_dimension, active_head_num, active_head_dim, active_modality:list, 
                  active_cross:list, active_cross_output:list):
        #print(active_self_attn_layer_num, active_hybrid_attn_layer_num, active_dimension, active_head_num, active_head_dim, active_modality)
        print('active modality in set active', active_modality, 'active cross', active_cross, 'active cross output', active_cross_output)
        self.active_modality = active_modality
        self.active_cross_output = active_cross_output
        self.active_cross = active_cross
        
        for k in self.trans_mems0.keys():
            self.trans_mems0[k].set_active(active_layer_num = active_hybrid_attn_layer_num, 
                                          active_dimension = active_dimension, 
                                          active_head_num = active_head_num, 
                                          active_head_dim = active_head_dim)
        for k in self.trans.keys():                                  
            self.trans[k].set_active(active_layer_num = active_hybrid_attn_layer_num, 
                                          active_dimension = active_dimension, 
                                          active_head_num = active_head_num, 
                                          active_head_dim = active_head_dim)
        # trans and trans_mem have different input dimension 
        for k in self.trans_mems.keys():                                  
            self.trans_mems[k].set_active(active_layer_num = active_self_attn_layer_num, 
                                      active_dimension = active_dimension, 
                                      active_head_num = active_head_num, 
                                      active_head_dim = active_head_dim)

    def sort(self, sort_head = False, sort_head_dim = False, sort_dim_transformer_layer = False):
        if sort_head:  
            print(' !!!!!!!!!!!!!!!!!! Sort head in each multihead attention transformer !!!!!!!!!!!!!!!')
        if sort_head_dim :
            print(' !!!!!!!!!!!!!!!!!! Sort hidden dimension of each head in each multihead attention transformer !!!!!!!!!!!!!!!')
        if sort_dim_transformer_layer:
            print(' !!!!!!!!!!!!!!!!!! Sort hidden dimension in last two linear layers for each transformer encoder layer !!!!!!!!!!!!!!!')
        for i in range(len(self.trans)):
            for j in range(len(self.trans[i])):
                self.trans[i][j].sort(sort_head, sort_head_dim, sort_dim_transformer_layer)
        for i in range(len(self.trans_mems)):
            self.trans_mems[i].sort(sort_head, sort_head_dim, sort_dim_transformer_layer)

    def gen_active_cross(self, active_modality:list, p_cross = 0.6, p_cross_output = 0.8):
        # the p is higher the number of active cross is larger
        active_cross = [[]] * self.modality_num
        active_cross_output = [[]] * self.modality_num
        if len(active_modality) == 1:
          active_cross[active_modality[0]] = []
          active_cross_output[active_modality[0]] = active_cross[active_modality[0]].copy() if active_cross[active_modality[0]] else [self.modality_list[active_modality[0]]]
          return active_cross, active_cross_output

        m = ModalityStr([self.modality_list[i] for i in active_modality])
        for i in active_modality:
            active_cross[i] = m.rand_gen_modality_str(modality_set = [self.modality_list[i]], p = p_cross)
            r = active_cross[i].copy()
            r = [self.modality_list[i]] + r
            active_cross_output[i] = gen_subnet(parent_set = r, p = p_cross_output)
            if not active_cross_output[i]:
                active_cross_output[i] = [active_cross[i][0] if active_cross[i] else self.modality_list[i]]
        return active_cross, active_cross_output
    
""" !!!!!!!!!!!!!!!! Test Code !!!!!!!!!!!!!!!! """
def count_parameters(model):
  parameter_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
  return parameter_num

import torch.optim as optim
import torch
import random
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np
import torchvision
from torch import nn
from torchvision import models
from src.dataset import *
from torch.utils.data import DataLoader
from src.eval_metrics import eval_mosei_senti

if __name__ == '__main__':
    torch.manual_seed(0)
    data = Multimodal_Datasets(dataset_path = '/content/drive/MyDrive/Colab_Notebooks/MultiBench-main/data', data='mosei_senti', split_type='train', if_align=True)
    train_data = data
    valid_data = Multimodal_Datasets(dataset_path = '/content/drive/MyDrive/Colab_Notebooks/MultiBench-main/data', data='mosei_senti', split_type='valid', if_align=True)
    
    train_loader = DataLoader(train_data, batch_size = 16, shuffle = True)
    valid_loader = DataLoader(valid_data, batch_size = 16, shuffle = True)

    dimension = 40
    num_heads = 8
    head_dim = 5
    layers_hybrid_attn = 4
    layers_self_attn = 3 

    m = DynamicMULTModel (origin_dimensions = [300, 74, 35, 35], dimension = dimension, 
        num_heads = num_heads, head_dim = head_dim, layers_hybrid_attn = layers_hybrid_attn, layers_self_attn = layers_self_attn, 
        attn_dropout = [0.1, 0, 0, 0, 0], relu_dropout = 0, res_dropout = 0, out_dropout = 0, embed_dropout = 0, 
        attn_mask = True, output_dim = 1, modality_set = ['t', 'a', 'v', 'V']).cuda()
    print(count_parameters(m) * 4/1024)

    modality_list = [[1, 2], [0, 1, 2], [0, 1, 2, 3]]

    active_cross = []
    active_cross_output = []

    optimizer = optim.Adam(m.parameters(), lr=0.001)
    eval_metric = nn.L1Loss()
    epoch = 25
    for i in range(epoch):
        m.eval()
        train_loss = 0
        for i_batch, (batch_X, batch_Y) in enumerate(train_loader): 
              print(i_batch)
              optimizer.zero_grad()
              sample_ind, text, audio, vision = batch_X
              x = [text.cuda(), audio.cuda(), vision.cuda(), vision.cuda()]
              result = m(x)
              #print('reach 0')
              if i_batch > 0:
                  result1 = s([x[i] for i in active_modality])
                  print(i_batch, result.mean().item(), result1.mean().item())
              loss = eval_metric(result, batch_Y.squeeze(-1).cuda())
              '''loss.backward()
              train_loss += loss.item()
              optimizer.step()'''
              
              #print('reach 1')
              active_self_attn_layer_num = torch.randint(low = 1, high = layers_self_attn + 1, size = (1, ))[0].item()
              active_hybrid_attn_layer_num = torch.randint(low = 1, high = layers_hybrid_attn + 1, size = (1, ))[0].item()
              active_dimension = torch.randint(low=1, high = dimension + 1, size = (1, ))[0].item()
              active_head_num = torch.randint(low=1, high = num_heads + 1, size = (1, ))[0].item()
              active_head_dim = torch.randint(low=1, high = head_dim + 1, size = (1, ))[0].item()
              active_modality = modality_list[torch.randint(low=0, high = len(modality_list), size = (1, ))[0].item()]
              #print('reach 1.0')
              active_cross, active_cross_output = m.gen_active_cross(active_modality)
              #print('reach 2')
              m.set_active(active_self_attn_layer_num = active_self_attn_layer_num, 
                          active_hybrid_attn_layer_num = active_hybrid_attn_layer_num, 
                          active_dimension = active_dimension, 
                          active_head_num = active_head_num,
                          active_head_dim = active_head_dim, 
                          active_modality = active_modality,
                          active_cross = active_cross, 
                          active_cross_output = active_cross_output)
              #print('reach 3')
              s = m.get_active_subnet(active_self_attn_layer_num = active_self_attn_layer_num, 
                              active_hybrid_attn_layer_num = active_hybrid_attn_layer_num, 
                              active_dimension = active_dimension, 
                              active_head_num = active_head_num, 
                              active_head_dim = active_head_dim , 
                              active_modality = active_modality,
                              active_cross = active_cross, 
                              active_cross_output = active_cross_output)
              s.eval()

              
        print('train loss: ', train_loss/len(train_loader))
        m.eval()
        results = []
        truths = []
        
        for i_batch, (batch_X, batch_Y) in enumerate(valid_loader):
              sample_ind, text, audio, vision = batch_X
              x = [text.cuda(), audio.cuda(), vision.cuda()]
              m.set_active(active_self_attn_layer_num = layers_self_attn, 
                          active_hybrid_attn_layer_num = layers_hybrid_attn, 
                          active_dimension = dimension , 
                          active_head_num = num_heads ,
                          active_head_dim =  head_dim , 
                          active_modality = [0, 1, 2])
              result = m(x)
              results.append(result.cpu().detach())
              truths.append(batch_Y.squeeze(-1).cpu().detach())
        results = torch.cat(results)
        truths = torch.cat(truths)
        eval_mosei_senti(results, truths)
        
     
    