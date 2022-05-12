import torch
from torch import nn
import torch.nn.functional as F

from modules.transformer import TransformerEncoder
from modules.dynamic_transformer import DynamicTransformerEncoder


def Amn(m, n):
    result = 1
    for i in range(m, m - n, -1):
      result *= i
    return result

def AmnSum(m):
    result = 0
    for n in range(1, m + 1):
        result += Amn(m, n)
    return result

class ModalityStr():
  """
  modality_set: list of single characters represent the modality name
  """
  def __init__(self, modality_set:list):
      self.modality_set = modality_set

  def gen_modality_str(self, input_str):
    modality_str = []
    for ch in self.modality_set:
      if input_str.find(ch) == -1:
        modality_str.append(input_str + ch)
    return modality_str

  def rand_gen_modality_str(self, modality_set:list, p = 0.5):
      modality_str = []
      assert not (len(modality_set) == len(self.modality_set) == 1)
      input_str1 = modality_set.copy()
      step = 1
      while step <= len(self.modality_set):
          input_str = []
          for s in input_str1:
              s_temp = self.gen_modality_str(s)
              probs = torch.rand(len(s_temp))
              s1 = [s_temp[i] for i in range(len(s_temp)) if probs[i] < p]
              modality_str.extend(s1)
              input_str.extend(s1)
          input_str1 = input_str
          step += 1
      return modality_str
  def gen_modality_str_all(self, modality_set:list = None):
    """
    generate all possible cross attention combinations 
    modality_set: modalities used to generate new modalities with self.modality_set
    index_list: index of cross attention combinations
    """
    modality_str = []
    if modality_set is None:
      input_str1 = self.modality_set.copy()
    else:
      assert not len(modality_set) == len(self.modality_set) == 1
      input_str1 = modality_set.copy()
    while len(modality_str) == 0 or len(modality_str[-1]) < len(self.modality_set):
      input_str = []
      for s in input_str1:
        s1 = self.gen_modality_str(s)
        modality_str.extend(s1)
        input_str.extend(s1)
      input_str1 = input_str
    return modality_str

def gen_subnet(parent_set:list, p):
    result = []
    probs = torch.rand((len(parent_set),))
    for i in range(len(probs)):
      if probs[i] < p:
        result.append(parent_set[i])
    return result

class MULTModel(nn.Module):
    def __init__(self, proj, trans_mems0, trans, trans_mems, proj1, proj2, out_layer,
        origin_dimensions:list, dimension, 
        num_heads, head_dim, layers_hybrid_attn, layers_self_attn, attn_dropout:list, 
        relu_dropout, res_dropout, out_dropout, embed_dropout, attn_mask, output_dim,
        cross, cross_output, modality_list, all_steps):
        super(MULTModel, self).__init__()

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
        self.all_steps = all_steps
        
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
        self.proj = proj

        """ Self Attentions (Shrinkable) """
        self.trans_mems0 = trans_mems0

        """ Crossmodal Attentions (Shrinkable) """
        self.trans = trans

        """ Self Attentions (Shrinkable) """
        self.trans_mems = trans_mems
         
        """ Projection Layers (Shrinkable) """
        self.proj1 = proj1
        self.proj2 = proj2
        self.out_layer = out_layer

        self.cross = cross
        self.cross_output = cross_output
        self.modality_list = modality_list
      
   
    """To be implemented! """
    def forward(self, x):
        assert len(x) == self.modality_num # missing modality will be repalced by ones or zeros, can not be deleted
        proj_x = [self.proj[i](x[i]) for i in range(self.modality_num)]
        proj_x = torch.stack(proj_x)
        proj_x = proj_x.permute(0, 3, 1, 2)

        proj_x1 = {self.modality_list[i]: self.trans_mems0['mems0' + self.modality_list[i]](proj_x[i], proj_x[i], proj_x[i]) for i in range(self.modality_num)}
        _h = proj_x1
        
        last_hs = []
        hs = []
        for i in range(self.modality_num):
            for m_c in self.cross[i]:
                _h[m_c] = self.trans['cross' + m_c]( _h[m_c[-1]], _h[m_c[:-1]], _h[m_c[:-1]])
            h = torch.cat([_h[m] for m in self.cross_output[i]], dim = 2)
            h = self.trans_mems['mems' + self.modality_list[i]](h)
            
            if self.all_steps:
                hs.append(h)
            else:
                last_hs.append(h[-1])
        
        if self.all_steps:
            out = torch.cat(hs, dim=2)  # [seq_len, batch_size, out_features]
            out = out.permute(1, 0, 2)  # [batch_size, seq_len, out_features]
        else:
            out = torch.cat(last_hs, dim=1)
        
        out_proj = self.proj2(
            F.dropout(F.relu(self.proj1(out)), p = self.out_dropout, training = self.training)
        )
        out_proj += out
        
        out = self.out_layer(out_proj)
        return out

