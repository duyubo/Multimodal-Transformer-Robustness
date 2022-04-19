import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
import sys

__all__ = ["MultiheadAttention"]

# multi head sttention based model, init directly with layers not by configuration
class MultiheadAttention(nn.Module):
    def __init__(self, in_proj_weight, in_proj_bias, out_proj, embed_dim_in, head_dim, num_heads, attn_dropout = 0.):
        super(MultiheadAttention, self).__init__()
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
        # W_q, W_k, W_v
        self.in_proj_weight = in_proj_weight
        #b_q, b_k, b_v
        self.in_proj_bias = in_proj_bias
        #out linear layer
        self.out_proj = out_proj

        """ TO be double checked !!! """
        assert self.out_proj.weight.data.size()[0] == self.embed_dim_out
        assert self.out_proj.weight.data.size()[1] == self.embed_dim
        assert 3 * self.embed_dim == in_proj_weight.size()[0]
        assert self.embed_dim_in == in_proj_weight.size()[1]

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
        attn = self.out_proj(attn)

        return attn

    def in_proj_qkv(self, query):
        return self._in_proj(query).chunk(3, dim=-1)

    def in_proj_kv(self, key):
        return self._in_proj(key, start = self.embed_dim).chunk(2, dim=-1)

    def in_proj_q(self, query, **kwargs):
        return self._in_proj(query, end = self.embed_dim, **kwargs)

    def in_proj_k(self, key):
        return self._in_proj(key, start = self.embed_dim, end=2 * self.embed_dim)

    def in_proj_v(self, value):
        return self._in_proj(value, start = 2 * self.embed_dim)

    def _in_proj(self, input, start=0, end=None, **kwargs):
        weight = kwargs.get('weight', self.in_proj_weight)
        bias = kwargs.get('bias', self.in_proj_bias)
        weight = weight[start:end, :]
        if bias is not None:
            bias = bias[start:end]
        return F.linear(input, weight, bias)