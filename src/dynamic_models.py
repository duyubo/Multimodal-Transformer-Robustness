import torch
from torch import nn
import torch.nn.functional as F
import sys
"""!!!!!!!!!!!!!!!!!!!!!!! To be deleted !!!!!!!!!!!!!!!!!!!!!!!"""
sys.path.append('/content/drive/MyDrive/Colab_Notebooks/Multimodal-Transformer-Robustness/')
"""!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"""

from modules.transformer import TransformerEncoder
from modules.dynamic_transformer import DynamicTransformerEncoder
from src.models import  MULTModel
from modules.dynamic_layers import DynamicLinear, DynamicLayerNorm

class DynamicMULTModel(MULTModel):
    def __init__(self, origin_dimensions:list, dimension, 
        num_heads, head_dim, layers_hybrid_attn, layers_self_attn, attn_dropout:list, 
        relu_dropout, res_dropout, out_dropout, embed_dropout, attn_mask, output_dim):
        
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

        """ Shrinkable Hyperparameters
        in the parent graph we set up each modality with the same number of layers, hidden dimensions, head numbers and etc. 
        But the number of layers, head numbers, head dim for each modality are not required to be same during sampling!
        Output dimension of the temporal conv layers should always be the same """
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.layers_hybrid_attn = layers_hybrid_attn
        self.layers_self_attn = layers_self_attn
        self.modality_num = len(self.orig_dimensions)
        self.combined_dim = self.modality_num * self.modality_num * self.d
        
        """ Temporal Convolutional Layers (None Shrinkable) """
        self.proj = [nn.Conv1d(self.orig_dimensions[i], self.d, kernel_size=3, padding=0, bias=False) for i in range(self.modality_num)]
        self.proj = nn.ModuleList(self.proj)

        """ Crossmodal Attentions (Shrinkable) """
        self.trans = [nn.ModuleList([self.get_network(i, j, mem=False, layers = self.layers_hybrid_attn) for j in range(self.modality_num)]) for i in range(self.modality_num)]
        self.trans = nn.ModuleList(self.trans)

        """ Self Attentions (Shrinkable) """
        self.trans_mems = [self.get_network(i, i, mem=True, layers=self.layers_self_attn) for i in range(self.modality_num)]
        self.trans_mems = nn.ModuleList(self.trans_mems)
         
        """ Projection Layers (Shrinkable) """
        self.proj1 = DynamicLinear(self.combined_dim, self.combined_dim, bias = True)
        self.proj2 = DynamicLinear(self.combined_dim, self.combined_dim, bias = True)
        self.out_layer = DynamicLinear(self.combined_dim, self.output_dim, bias = True)

        super(DynamicMULTModel, self).__init__(
            proj = self.proj, trans = self.trans, trans_mems = self.trans_mems, 
            proj1 = self.proj1, proj2 = self.proj2, out_layer = self.out_layer,
            origin_dimensions = self.orig_dimensions, dimension = self.d, 
            num_heads = self.num_heads, head_dim = self.head_dim, layers_hybrid_attn = self.layers_hybrid_attn,
            layers_self_attn = self.layers_self_attn, attn_dropout = self.attn_dropout, 
            relu_dropout = self.relu_dropout, res_dropout = self.res_dropout, out_dropout = self.out_dropout, 
            embed_dropout = self.embed_dropout, attn_mask = self.attn_mask, output_dim = self.output_dim)

        self.active_modality = list(range(self.modality_num)) # saving index of the modality

    def get_network(self, mod1, mod2, mem, layers=-1):
        if not mem:
            embed_dim_in = self.d
            attn_dropout = self.attn_dropout[mod1]
        else:
            embed_dim_in = self.modality_num * self.d
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
        #print('active modality in forward', self.active_modality)
        proj_x = [self.proj[i](x[i]) for i in self.active_modality]
        proj_x = torch.stack(proj_x)
        proj_x = proj_x.permute(0, 3, 1, 2)
        
        modality_len = len(self.active_modality)
                
        hs = []
        last_hs = []
        for i in range(modality_len):
            h = []
            for j in range(modality_len):
                h.append(self.trans[self.active_modality[i]][self.active_modality[j]](proj_x[i], proj_x[j], proj_x[j]))
            h = torch.cat(h, dim = 2)
           
            """ TO be Tested! """
            if not len(self.active_modality) == self.modality_num:
                active_mask = []
                for jj in self.active_modality:
                  active_mask.extend(list(range(jj * self.d, jj * self.d + self.d)))
                active_mask = torch.tensor(active_mask).type(torch.IntTensor).to(next(self.parameters()).device)
            else:
                active_mask = [None]
            h = self.trans_mems[self.active_modality[i]](h, active_mask = active_mask)
            if type(h) == tuple:
                h = h[0]
            last_hs.append(h[-1])
        
        out = torch.cat(last_hs, dim=1)
        
        if not modality_len == self.modality_num:
            active_indexes = []
            for i in self.active_modality:
              for j in self.active_modality:
                active_indexes.extend(list(range((i * self.modality_num + j) * self.d, (i * self.modality_num + j + 1) * self.d)))
            active_indexes = torch.Tensor(active_indexes).type(torch.IntTensor).to(next(self.parameters()).device)
        else:
            active_indexes = [None]    
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

    """To be Modified -> each modality has different settings"""
    def get_active_subnet(self, active_self_attn_layer_num, 
                          active_hybrid_attn_layer_num, 
                          active_dimension, 
                          active_head_num, 
                          active_head_dim, 
                          active_modality:list):
        #print(active_self_attn_layer_num, active_hybrid_attn_layer_num, active_dimension, active_head_num , active_head_dim , active_modality)
        #print('active modality in get_active_subnet', active_modality)
        proj = []
        for i in active_modality:
          p = nn.Conv1d(self.orig_dimensions[i], self.d, kernel_size=3, padding=0, bias=False)
          p.weight.data.copy_(self.proj[i].weight.data)
          proj.append(p)

        trans = [nn.ModuleList([self.trans[j][i].get_active_subnet(
                                                  active_layer_num = active_hybrid_attn_layer_num, 
                                                  active_dimension = active_dimension, 
                                                  active_head_num = active_head_num, 
                                                  active_head_dim = active_head_dim, 
                                                  active_mask = [None]
                                                  ) 
                                for i in active_modality]
                              )
                for j in active_modality]

        trans_mems = []
        for i in active_modality:
            active_mask = []
            for jj in active_modality:
              active_mask.extend(list(range(jj * self.d, jj * self.d + self.d)))
            active_mask = torch.Tensor(active_mask).type(torch.IntTensor).to(next(self.parameters()).device)
            trans_mems.append(self.trans_mems[i].get_active_subnet(
                                                  active_layer_num = active_self_attn_layer_num, 
                                                  active_dimension = active_dimension, 
                                                  active_head_num = active_head_num, 
                                                  active_head_dim = active_head_dim, 
                                                  active_mask = active_mask))
        
        active_indexes = []
        for i in active_modality:
          for j in active_modality:
            active_indexes.extend(
              list(
                range(
                  (i * self.modality_num + j) * self.d, (i * self.modality_num + j + 1) * self.d
                )
              )
            )
        active_indexes = torch.Tensor(active_indexes).type(torch.IntTensor).to(next(self.parameters()).device)

        proj1 = self.proj1.copy(dim_in = None, dim_out = None, mask_in = active_indexes, mask_out = [None])
        proj2 = self.proj2.copy(dim_in = None, dim_out = None, mask_in = [None], mask_out = active_indexes)
        out_layer = self.out_layer.copy(dim_in = None, dim_out = None, mask_in = active_indexes, mask_out = [None])

        attn_drop = [self.attn_dropout[i] for i in active_modality]
        attn_drop.append(self.attn_dropout[-1])
        model = MULTModel(
            proj = proj, trans = trans, trans_mems = trans_mems, 
            proj1 = proj1, proj2 = proj2, out_layer = out_layer,
            origin_dimensions = [self.orig_dimensions[i] for i in active_modality], dimension = self.d, 
            num_heads = active_head_num, head_dim = active_head_dim, layers_hybrid_attn = active_hybrid_attn_layer_num,
            layers_self_attn = active_self_attn_layer_num, attn_dropout = attn_drop, 
            relu_dropout = self.relu_dropout, res_dropout = self.res_dropout, out_dropout = self.out_dropout, 
            embed_dropout = self.embed_dropout, attn_mask = self.attn_mask, output_dim = self.output_dim)
        
        model = model.to(self.parameters().__next__().device)
        return model

    def set_active(self, active_self_attn_layer_num, active_hybrid_attn_layer_num, active_dimension, active_head_num, active_head_dim, active_modality:list):
        #print(active_self_attn_layer_num, active_hybrid_attn_layer_num, active_dimension, active_head_num, active_head_dim, active_modality)
        self.active_modality = active_modality
        for i in active_modality:
          for j in active_modality:
              self.trans[i][j].set_active(active_layer_num = active_hybrid_attn_layer_num, 
                                          active_dimension = active_dimension, 
                                          active_head_num = active_head_num, 
                                          active_head_dim = active_head_dim)
          # trans and trans_mem have different input dimension 
          self.trans_mems[i].set_active(active_layer_num = active_self_attn_layer_num, 
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
            for j in range(len(self.trans[0])):
                self.trans[i][j].sort(sort_head, sort_head_dim, sort_dim_transformer_layer)
        for i in range(len(self.trans_mems)):
            self.trans_mems[i].sort(sort_head, sort_head_dim, sort_dim_transformer_layer)

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

    dimension = 30
    num_heads = 8
    head_dim = 5
    layers_hybrid_attn = 4
    layers_self_attn = 3 

    m = DynamicMULTModel (origin_dimensions = [300, 74, 35], dimension = 30, 
        num_heads = 8, head_dim = 5, layers_hybrid_attn = 4, layers_self_attn = 3, attn_dropout = [0.1, 0.1, 0.1, 0], 
        relu_dropout = 0, res_dropout = 0, out_dropout = 0, embed_dropout = 0, attn_mask = True, output_dim = 1).cuda()
    print(count_parameters(m) * 4/1024)
    modality_list = [[0], [1], [2], [0, 1], [0, 2], [1, 2], [0, 1, 2]]

    optimizer = optim.Adam(m.parameters(), lr=0.001)
    eval_metric = nn.L1Loss()
    epoch = 25
    for i in range(epoch):
        m.train()
        train_loss = 0
        for i_batch, (batch_X, batch_Y) in enumerate(train_loader): 
              optimizer.zero_grad()
              sample_ind, text, audio, vision = batch_X
              x = [text.cuda(), audio.cuda(), vision.cuda()]
              result = m(x)
              if i_batch > 0:
                  result1 = s([x[i] for i in active_modality])
                  '''if result.mean().item() != result1.mean().item():
                      print(i_batch, result.mean().item(), result1.mean().item())'''
              loss = eval_metric(result, batch_Y.squeeze(-1).cuda())
              loss.backward()
              train_loss += loss.item()
              optimizer.step()

              active_self_attn_layer_num = torch.randint(low = 1, high = layers_self_attn + 1, size = (1, ))[0].item()
              active_hybrid_attn_layer_num = torch.randint(low = 1, high = layers_hybrid_attn + 1, size = (1, ))[0].item()
              active_dimension = torch.randint(low=1, high = dimension + 1, size = (1, ))[0].item()
              active_head_num = torch.randint(low=1, high = num_heads + 1, size = (1, ))[0].item()
              active_head_dim = torch.randint(low=1, high = head_dim + 1, size = (1, ))[0].item()
              active_modality = modality_list[torch.randint(low=0, high = len(modality_list), size = (1, ))[0].item()]
              
              m.set_active(active_self_attn_layer_num = active_self_attn_layer_num, 
                          active_hybrid_attn_layer_num = active_hybrid_attn_layer_num, 
                          active_dimension = active_dimension, 
                          active_head_num = active_head_num,
                          active_head_dim = active_head_dim, 
                          active_modality = active_modality)
              
              s = m.get_active_subnet(active_self_attn_layer_num = active_self_attn_layer_num, 
                              active_hybrid_attn_layer_num = active_hybrid_attn_layer_num, 
                              active_dimension = active_dimension, 
                              active_head_num = active_head_num, 
                              active_head_dim = active_head_dim , 
                              active_modality = active_modality)
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
        
     
    