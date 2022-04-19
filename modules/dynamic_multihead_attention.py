import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
import sys
import warnings

"""!!!!!!!!!!!!!!!!!!!!!!! To be deleted !!!!!!!!!!!!!!!!!!!!!!!"""
sys.path.append('/content/drive/MyDrive/Colab_Notebooks/Multimodal-Transformer-master/')
"""!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"""

from modules.multihead_attention import *
import functools

__all__ = ["DynamicMultiheadAttention"]

class DynamicMultiheadAttention(MultiheadAttention):
    """
    Modified from source code of 
    Multi-headed attention.
    See "Attention Is All You Need" for more details.
    """
    def __init__(self, embed_dim_in, head_dim, num_heads, attn_dropout=0.):
        nn.Module.__init__(self)
        # dim of input q, k, v
        self.embed_dim_in = embed_dim_in
        # we keep the input and output dimension as fixed
        self.embed_dim_out = self.embed_dim_in
        # output dim of W^qq, W^kk, W^vv
        self.embed_dim = head_dim * num_heads
        # number of heads (multihead attention module)
        self.num_heads = num_heads
        self.attn_dropout = attn_dropout
        # dim of a single head
        self.head_dim = head_dim
        # W_q, W_k, W_v [out_features, in_features]
        self.in_proj_weight = Parameter(torch.Tensor(3 * self.embed_dim, self.embed_dim_in))
        # b_q, b_k, b_v [out_features]
        self.in_proj_bias = Parameter(torch.Tensor(3 * self.embed_dim))
        # convert the output dimension back to the input diemnsion for residule connection in the future
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim_out, bias=True)
        # init weights
        super(DynamicMultiheadAttention, self).__init__(
            self.in_proj_weight, self.in_proj_bias, self.out_proj, self.embed_dim_in, self.head_dim, self.num_heads, self.attn_dropout  
        )
        # active number of heads and hidden dimensions for each head
        # will be used during inference
        self.active_head_dim = self.head_dim
        self.active_num_heads = self.num_heads

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.constant_(self.in_proj_bias, 0.)
        nn.init.constant_(self.out_proj.bias, 0.)

    """Tested!"""
    def forward(self, query, key, value, attn_mask = None, active_mask = [None]):
        """Input shape: Time x Batch x Channel
        Self-attention can be implemented by passing in the same arguments for
        query, key and value. Timesteps can be masked by supplying a T x T mask in the
        `attn_mask` argument. Padding elements can be excluded from
        the key by passing a binary ByteTensor (`key_padding_mask`) with shape:
        batch x src_len, where padding elements are indicated by 1s.
        """
        qkv_same = query.data_ptr() == key.data_ptr() == value.data_ptr()
        kv_same = key.data_ptr() == value.data_ptr()

        tgt_len, bsz, embed_dim = query.size()
        #assert embed_dim == self.embed_dim_in
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        assert key.size() == value.size()

        aved_state = None

        if qkv_same:
            """ self-attention """
            q, k, v = self.in_proj_qkv(query, active_mask = active_mask)
        elif kv_same:
            """ encoder-decoder attention """
            assert active_mask[0] is None  # assume input dimension slice only happens when self attention
            q = self.in_proj_q(query)
            if key is None:
                assert value is None
                k = v = None
            else:
                k, v = self.in_proj_kv(key)
        else:
            q = self.in_proj_q(query)
            k = self.in_proj_k(key)
            v = self.in_proj_v(value)
             
        q = q * (self.active_head_dim ** -0.5)

        q = q.contiguous().view(tgt_len, bsz * self.active_num_heads, self.active_head_dim).transpose(0, 1)
        k = k.contiguous().view(-1, bsz * self.active_num_heads, self.active_head_dim).transpose(0, 1)
        v = v.contiguous().view(-1, bsz * self.active_num_heads, self.active_head_dim).transpose(0, 1)
        #print('parent graph after qkv: ', q.mean().item(), k.mean().item(), v.mean().item())
        src_len = k.size(1)

        attn_weights = torch.bmm(q, k.transpose(1, 2))
        assert list(attn_weights.size()) == [bsz * self.active_num_heads, tgt_len, src_len]
        
        try:
            attn_weights += attn_mask.unsqueeze(0)
        except:
            print(attn_weights.shape)
            print(attn_mask.unsqueeze(0).shape)
            assert False
               
        attn_weights = F.softmax(attn_weights.float(), dim = -1).type_as(attn_weights)
        attn_weights = F.dropout(attn_weights, p = self.attn_dropout, training = self.training)
        # attn_weights.size() == [bsz * self.active_num_heads, tgt_len, src_len]
        # v.size() == [bsz * self.active_num_heads, src_len, self.active_head_dim]
        attn = torch.bmm(attn_weights, v)
        assert list(attn.size()) == [bsz * self.active_num_heads, tgt_len, self.active_head_dim]
        
        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, self.active_num_heads * self.active_head_dim)
        attn = self._out_proj(attn, active_mask = active_mask)

        return attn

    """ To be Tested !!!!!! """ 
    def get_active_subnet(self, active_head_dim, active_num_heads, active_mask = [None]):
        """
        get_active_subnet selects an active subnet 
        Arguments:
          embed_dim_per_head_active: selected embed dimension in each head 
          num_heads_active: number of heads selected
          return: a new subnet MultiheadAttention module with exactly the same weight as current module

        .. note:: 
          the selected active head number and embeded dimension of each head are the same for W_q, W_k, W_v
          we assume the weights are already sorted
        """ 
        # weight.size() = [out_features, in_features]
        in_proj_weight = self.in_proj_weight.data.contiguous().view(3, self.num_heads, self.head_dim, self.embed_dim_in)
        in_proj_weight = in_proj_weight[:, :active_num_heads, :active_head_dim, :]
        if not active_mask[0] is None:
          in_proj_weight = torch.index_select(in_proj_weight, -1, active_mask)
        in_proj_weight = in_proj_weight.contiguous().view(3 * active_num_heads * active_head_dim, -1)

        #bias.size() = [out_features]
        in_proj_bias = self.in_proj_bias.data.contiguous().view(3, self.num_heads, self.head_dim)
        in_proj_bias = in_proj_bias[:, :active_num_heads, :active_head_dim]
        in_proj_bias = in_proj_bias.contiguous().view(-1)

        out_proj = nn.Linear(active_num_heads * active_head_dim, self.embed_dim_out, bias=True).to(self.parameters().__next__().device)
        out_proj.weight.data.copy_((self.out_proj.weight.data.contiguous().view(-1, self.num_heads, self.head_dim)[:, : active_num_heads, :active_head_dim]).contiguous().view(-1, active_num_heads * active_head_dim))
        out_proj.bias.data.copy_(self.out_proj.bias.data)
        if not active_mask[0] is None:
          out_proj.weight.data = torch.index_select(out_proj.weight.data, 0, active_mask)
          out_proj.bias.data = torch.index_select(out_proj.bias.data, 0, active_mask)
        
        active_emded_dim_in = len(active_mask) if (not active_mask[0] is None) else self.embed_dim_in
        sub_layer = MultiheadAttention(in_proj_weight = Parameter(in_proj_weight), 
                                        in_proj_bias = Parameter(in_proj_bias), 
                                        out_proj = out_proj, 
                                        embed_dim_in = active_emded_dim_in, 
                                        head_dim = active_head_dim, 
                                        num_heads = active_num_heads, 
                                        attn_dropout = self.attn_dropout)
        sub_layer = sub_layer.to(self.parameters().__next__().device)

        return sub_layer
    
    """ Tested"""
    def sort_hidden_layers(self):
        """
        Sort the hidden layers for each head or heads with importance and return the importance index (descending order)
        Arguments:
          dimension: dimension of head or hidden layers

        ..note:: 
          sort in_proj_weight and in_proj_bias, we do not sort out_proj
          reference "https://github.com/mit-han-lab/once-for-all/blob/master/ofa/imagenet_classification/elastic_nn/modules/dynamic_layers.py"
        
        """
        dimension = 2
        dimension_list = [3, self.num_heads, self.head_dim, self.embed_dim_in]
        importance = torch.sum(
            torch.abs(self.in_proj_weight.data.view(dimension_list)), 
            dim = (0, 3),
            keepdim = False
        )
        sorted_importance, sorted_idx = torch.sort(importance, dim = 1, descending=True)
        renormalized_index = torch.ones([3, self.num_heads, self.head_dim])
        renormalized_index_out = torch.ones([self.num_heads, self.head_dim])
        for i in range(3):
          for j in range(self.num_heads):
            base = i * self.num_heads * self.head_dim  + j * self.head_dim + sorted_idx[j][:]
            renormalized_index[i, j] = base
            renormalized_index_out[j] = j * self.head_dim + sorted_idx[j][:]
        selected_indexes = torch.Tensor(renormalized_index).view(-1).type(torch.IntTensor).to(self.parameters().__next__().device)
        selected_indexes_out = torch.Tensor(renormalized_index_out).view(-1).type(torch.IntTensor).to(self.parameters().__next__().device)
        """sort out dimension of in_proj_weight"""
        self.in_proj_weight.data = torch.index_select(
            self.in_proj_weight.data, 0, selected_indexes
        ) 
        """sort out dimension of in_proj_bias with the same index as in_proj_weight"""
        self.in_proj_bias.data = torch.index_select(
            self.in_proj_bias.data, 0, selected_indexes
        )
        """sort in dimension of the out_proj layer"""
        self.out_proj.weight.data = torch.index_select(
             self.out_proj.weight.data, 1, selected_indexes_out
        )
        return sorted_idx.tolist()
    
    """ TO be Tested !!!!!!"""
    def sort_heads(self):
        """sort heads by their importance"""
        dimension_list = [3, self.num_heads, self.head_dim, self.embed_dim_in]
        importance = torch.sum(
            torch.abs(self.in_proj_weight.data.view(dimension_list)), 
            dim = (0, 2, 3),
            keepdim = False
        )
        sorted_importance, sorted_idx = torch.sort(importance, dim = 0, descending=True)
        #print(sorted_idx)
        renormalized_index =[]
        renormalized_index_out =[]
        for i in range(3):
          for j in range(self.num_heads):
            base = i * self.num_heads * self.head_dim  +  self.head_dim * sorted_idx[j] 
            renormalized_index.append(range(base, base + self.head_dim))
            if i == 0:
              renormalized_index_out.append(range(sorted_idx[j] * head_dim, sorted_idx[j] * head_dim + head_dim))
        selected_indexes = torch.Tensor(renormalized_index).view(-1).type(torch.IntTensor).to(self.parameters().__next__().device)
        selected_indexes_out = torch.Tensor(renormalized_index_out).view(-1).type(torch.IntTensor).to(self.parameters().__next__().device)
        """sort in_proj_weight"""
        self.in_proj_weight.data = torch.index_select(
            self.in_proj_weight.data, 0, selected_indexes
        ) 
        """sort in_proj_bias with the same index as in_proj_weight"""
        self.in_proj_bias.data = torch.index_select(
            self.in_proj_bias.data, 0, selected_indexes
        )
        """sort in dimension of the out_proj layer"""
        self.out_proj.weight.data = torch.index_select(
             self.out_proj.weight.data, 1, selected_indexes_out
        )
        return sorted_idx.tolist()
         
    def in_proj_qkv(self, query, active_mask = [None]):
        return self._in_proj(query, 0, 3, active_mask = active_mask).chunk(3, dim=-1)

    def in_proj_kv(self, key, active_mask = None):
        return self._in_proj(key, 1, 3).chunk(2, dim=-1)

    def in_proj_q(self, query, active_mask = None):
        return self._in_proj(query, 0, 1)

    def in_proj_k(self, key, active_mask = None):
        return self._in_proj(key, 1, 2)

    def in_proj_v(self, value, active_mask = None):
        return self._in_proj(value, 2, 3)
    
    """To be Tested!"""
    def _in_proj(self, input, start, end, active_mask = [None]):
        dimension_list = [3, self.num_heads, self.head_dim, self.embed_dim_in]
        weight = (self.in_proj_weight.contiguous().view(dimension_list)[start:end, :self.active_num_heads , :self.active_head_dim, :self.embed_dim_in]).reshape((end - start) * self.active_num_heads * self.active_head_dim, self.embed_dim_in)
        bias = (self.in_proj_bias.contiguous().view(dimension_list[:-1])[start:end, :self.active_num_heads , :self.active_head_dim]).reshape(-1)
        """input dimension of _in_proj will be decided only by active_mask"""
        if not active_mask[0] is None:
            weight = torch.index_select(
                weight, -1, active_mask
            )
        return F.linear(input, weight, bias)
    
    """To be Tested!"""
    def _out_proj(self, input, active_mask = [None]):
        weight = (self.out_proj.weight.contiguous().view(self.embed_dim_out, self.num_heads, self.head_dim)[:, :self.active_num_heads, :self.active_head_dim]).reshape(self.embed_dim_out, self.active_head_dim * self.active_num_heads)
        bias = self.out_proj.bias.contiguous()
        """output dimension of out_proj will be decided only by active_mask"""
        if not active_mask[0] is None:
          weight = torch.index_select(
            weight, 0, active_mask
          )
          bias = torch.index_select(
            bias, 0, active_mask
          )
        return F.linear(input, weight, bias)

    def set_active(self, active_head_dim, active_num_heads):
        self.active_head_dim = active_head_dim
        self.active_num_heads = active_num_heads

"""!!!!!!!!!!!!!!!!!!!!!!!!!!!! Test Module !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"""
def count_parameters(model):
  return sum(p.numel() for p in model.parameters() if p.requires_grad)

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

if __name__ == '__main__':
    torch.manual_seed(0)
    embed_dim_in = 20
    head_dim = 40
    num_heads = 30

    """ Test basic construct function"""
    dm = DynamicMultiheadAttention(embed_dim_in, head_dim, num_heads, attn_dropout=0.).cuda()
    t = torch.ones(30, 30).cuda()
    future_mask = torch.triu(t.float().fill_(float('-inf')).type_as(t), 1+abs(0))
    
    optimizer = optim.SGD(dm.parameters(), lr=1, momentum=0.9)
    
    loss_list = []
    loss_list1 = []
    order_change_list_hidden = []
    order_change_list_head = []
    order_list_hidden  = None
    order_list_head = None
    least_error = 100
    lr1 = 0.005
    total = 25000#10000
    for i in range(total):
      """training mode"""
      dm.train()
      """Test forward function"""
      x = torch.rand(30, 50, 20).cuda()
      """Test random sample subnet and train"""
      if i == 25000:
        optimizer.param_groups[0]['lr'] = lr1
        order_list_hidden_temp = dm.sort_hidden_layers()
        if not order_list_hidden == order_list_hidden_temp:
            order_list_hidden = order_list_hidden_temp
            order_change_list_hidden.append(i)
      if (i > 25000 and i < 40000):
        """Test reorder hidden dimensions"""
        #dm.active_num_heads = torch.randint(1, num_heads + 1, (1, ))[0].item()
        dm.active_head_dim = torch.randint(1, head_dim + 1, (1, ))[0].item()
      if (i == 10000):
        optimizer.param_groups[0]['lr'] = lr1
        order_list_head_temp = dm.sort_heads()
        if not order_list_head == order_list_head_temp:
            order_list_head = order_list_head_temp
            order_change_list_head.append(i)
      if (i > 10000):
        """Test reorder heads, Failed !!!!"""
        dm.active_head_dim = torch.randint(1, head_dim + 1, (1, ))[0].item()
        dm.active_num_heads = torch.randint(1, num_heads + 1, (1, ))[0].item()
      loss = abs(dm(x, x, x, future_mask) - (x.sin() * x.exp())).mean()
      print(i, 'active head dimensions: ', dm.active_head_dim, 
              'active number of heads: ', dm.active_num_heads, 
              "loss: ", loss.item())
      loss.backward()
      loss_list.append(loss.item())
      optimizer.step()
      dm.zero_grad()
      if loss < least_error:
        least_error = loss 

      """validation mode"""
      dm.eval()
      dm.active_head_dim = head_dim
      dm.active_num_heads = num_heads
      loss1 = abs(dm(x, x, x, future_mask) - (x.sin() * x.exp())).mean()
      loss_list1.append(loss1.item())

    x = torch.rand(30, 50, 20).cuda()
    """Test get active subnet"""
    dm.eval()
    r = torch.ones([num_heads * head_dim, ])
    for i in range(1, num_heads + 1):
      for j in range(1, head_dim + 1):
        num_heads_active = i
        embed_dim_per_head_active = j
        subnet = dm.get_active_subnet(j, i)
        subnet.eval()
        dm.active_num_heads = i
        dm.active_head_dim = j
        n1_loss = abs(subnet(x, x, x, future_mask) - (x.sin() * x.exp())).mean().item()
        r[(i - 1) * head_dim + j - 1] = n1_loss
        print('number heads: ', num_heads_active, 
              'dimensions per head: ', embed_dim_per_head_active,  
              'parameter number: ', count_parameters(subnet),
              n1_loss,
              abs(dm(x, x, x, future_mask) - (x.sin() * x.exp())).mean().item()
        )

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    Y = np.arange(1, num_heads + 1)
    X = np.arange(1, head_dim + 1)
    X, Y = np.meshgrid(X, Y)
    Z = np.array(r.tolist()).reshape(num_heads, head_dim)
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
    ax.set_ylabel('head number')
    ax.set_xlabel('dimension per head')
    ax.set_zlabel('L1 loss')
    ax.zaxis.set_major_locator(LinearLocator(10))
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.savefig('3d.png')
    plt.show()

    plt.rcParams["figure.figsize"] = (10,10)
    i = 0
    for o in order_change_list_hidden:
      if i == 0:
        plt.plot([o, o], [0.05, 0.4], color = 'purple', label = 'hidden layers order changed')
      else:
        plt.plot([o, o], [0.05, 0.4], color = 'purple')
      i = i + 1
    i = 0
    for o in order_change_list_head:
      if i == 0:
        plt.plot([o, o], [0.05, 0.4], color = 'green', label = 'heads order changed')
      else:
        plt.plot([o, o], [0.05, 0.4], color = 'green')
    plt.scatter(range(len(loss_list)), loss_list, label = "subnet", color = 'blue', s = 0.1)
    plt.plot(loss_list1, label = "parent net", color = 'orange', linewidth = 0.5)
    plt.plot([0, total], [least_error.item(), least_error.item()], color = 'red', label = 'lowest L1 Loss', linewidth = 0.5)
    plt.xlabel('training epoches')
    plt.ylabel('L1 Loss')
    plt.text(10000, 0.3, 'lr = %f'%lr1, color = 'red')
    plt.title('Test on toy model: training + sort heads + random head number \n sort hidden dimensions + random sample hidden dimensions ') #      + \n
    plt.legend()
    plt.savefig('training.png')  
