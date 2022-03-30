import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
import sys

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
        # sqrt(d_k)
        self.scaling = self.head_dim ** -0.5
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

    """To be Tested!"""
    def forward(self, query, key, value, attn_mask=None):
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
        assert embed_dim == self.embed_dim_in
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        assert key.size() == value.size()

        aved_state = None

        self.active_head_dim
        self.active_num_heads

        if qkv_same:
            # self-attention
            q, k, v = self.in_proj_qkv(query)
        elif kv_same:
            # encoder-decoder attention
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
        q = q * self.scaling

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        src_len = k.size(1)

        attn_weights = torch.bmm(q, k.transpose(1, 2))
        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        try:
            attn_weights += attn_mask.unsqueeze(0)
        except:
            print(attn_weights.shape)
            print(attn_mask.unsqueeze(0).shape)
            assert False
                
        attn_weights = F.softmax(attn_weights.float(), dim=-1).type_as(attn_weights)
        attn_weights = F.dropout(attn_weights, p=self.attn_dropout, training=self.training)

        attn = torch.bmm(attn_weights, v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]

        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, self.embed_dim)
        attn = self._out_proj(attn)

        return attn

    """ TO be Tested !!!!!! """ 
    def get_active_subnet(self, embed_dim_per_head_active, num_heads_active, **kwargs):
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
        weight = kwargs.get('weight', self.in_proj_weight)
        in_proj_weight = weight.contiguous().view(3, self.num_heads, self.head_dim, self.embed_dim_in)
        in_proj_weight = in_proj_weight[:, :num_heads_active, :embed_dim_per_head_active, :]
        in_proj_weight = in_proj_weight.contiguous().view(3* num_heads_active * embed_dim_per_head_active, self.embed_dim_in)

        #bias.size() = [out_features]
        bias = kwargs.get('bias', self.in_proj_bias)
        in_proj_bias = bias.contiguous().view(3, self.num_heads, self.head_dim)
        in_proj_bias = in_proj_bias[:, :num_heads_active, :embed_dim_per_head_active]
        in_proj_bias = in_proj_bias.contiguous().view(-1)

        out_proj = nn.Linear(num_heads_active * embed_dim_per_head_active, self.embed_dim_out, bias=True)
        out_proj.weight.data.copy_(self.out_proj.weight.data[:, : num_heads_active * embed_dim_per_head_active])
        out_proj.bias.data.copy_(self.out_proj.bias.data)

        sub_layer = MultiheadAttention(Parameter(in_proj_weight), Parameter(in_proj_bias), out_proj, self.embed_dim_in, embed_dim_per_head_active, num_heads_active, attn_dropout = self.attn_dropout)
        sub_layer = sub_layer.to(self.parameters().__next__().device)

        return sub_layer
    
    """ TO be Tested !!!!!!"""
    def _sort_dimension(self, dimension, total_dimension = 4):
        """
        Sort the hidden layers for each head or heads with importance and return the importance index (descending order)
        Arguments:
          dimension: dimension of head or hidden layers

        ..note:: 
          sort in_proj_weight and in_proj_bias, we do not sort out_proj
          reference "https://github.com/mit-han-lab/once-for-all/blob/master/ofa/imagenet_classification/elastic_nn/modules/dynamic_layers.py"
        
        """
        dimension_list = [3, self.num_heads, self.head_dim, self.embed_dim_in]
        importance = torch.sum(
            torch.abs(self.in_proj_weight.data.view(dimension_list)), dim = tuple(i for i in range(total_dimension) if i != dimension)
        )
        sorted_importance, sorted_idx = torch.sort(importance, dim = 0, descending=True)
        
        renormalized_index = functools.reduce(lambda x, y: x * y, dimension_list[ : dimension + 1])
        # sort in_proj_weight
        self.in_proj_weight.data = torch.index_select(
            self.in_proj_weight.data, 0, sorted_idx * renormalized_index
        ) 
        # sort in_proj_bias with the same index as in_proj_weight
        self.in_proj_bias.data = torch.index_select(
            self.in_proj_weight.bias, 0, sorted_idx * renormalized_index
        )
        return sorted_idx
    
    """ TO be Tested !!!!!!"""
    def sort_heads(self):
        """sort heads by their importance"""
        return self._sort_dimension(dimension = 1)
    """ TO be Tested !!!!!!"""
    def sort_hidden_layers(self):
        """sort hidden layers by their importance for each head"""    
        return self._sort_dimension(dimension = 2)

    def in_proj_qkv(self, query):
        return self._in_proj(query, 0, 3).chunk(3, dim=-1)

    def in_proj_kv(self, key):
        return self._in_proj(key, 1, 3).chunk(2, dim=-1)

    def in_proj_q(self, query, **kwargs):
        return self._in_proj(query, 0, 1, **kwargs)

    def in_proj_k(self, key):
        return self._in_proj(key, 1, 2)

    def in_proj_v(self, value):
        return self._in_proj(value, 2, 3)
    
    """To be Tested!"""
    def _in_proj(self, input, start, end, **kwargs):
        weight = kwargs.get('weight', self.in_proj_weight)
        bias = kwargs.get('bias', self.in_proj_bias)

        dimension_list = [3, self.num_heads, self.head_dim, self.embed_dim_in]
        weight = weight.contiguous().view(dimension_list)
        bias = bias.contiguous().view(dimension_list[:-1])

        weight = weight[start:end, :self.active_num_heads , :self.active_head_dim, :].view(3 * self.active_num_heads * self.active_head_dim, -1)
        bias = bias[start:end, :self.active_num_heads , :self.active_head_dim].view(-1)
        return F.linear(input, weight, bias)
    
    """To be Tested!"""
    def _out_proj(self, input, **kwargs):
        weight = self.out_proj.weight[:, :self.active_head_dim * self.active_num_heads]
        bias = self.out_proj.bias[:self.active_head_dim * self.active_num_heads]
        return F.linear(input, weight, bias)

