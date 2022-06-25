import sys

import torch
from torch import nn
import torch.nn.functional as F
from modules.position_embedding import SinusoidalPositionalEmbedding
from modules.multihead_attention import MultiheadAttention
from modules.dynamic_multihead_attention import DynamicMultiheadAttention
from modules.multihead_attention import MultiheadAttention
from modules.dynamic_layers import DynamicLinear, DynamicLayerNorm
import math
from modules.transformer import *



class DynamicTransformerEncoder(TransformerEncoder):
    def __init__(self, embed_dim, head_dim, num_heads, layers, attn_dropout=0.0, relu_dropout=0.0, res_dropout=0.0,
                 embed_dropout=0.0, attn_mask=False):
        nn.Module.__init__(self)
        
        self.dropout = embed_dropout      
        self.attn_dropout = attn_dropout
        self.relu_dropout = relu_dropout
        self.res_dropout = res_dropout
        self.embed_dropout = embed_dropout
        self.embed_dim = embed_dim
        self.embed_scale = math.sqrt(self.embed_dim)
        self.embed_positions = SinusoidalPositionalEmbedding(self.embed_dim)
        
        self.attn_mask = attn_mask

        self.layers = nn.ModuleList([])
        for layer in range(layers):
            new_layer = DynamicTransformerEncoderLayer(embed_dim_in = self.embed_dim, 
                                                head_dim = head_dim,
                                                num_heads = num_heads,
                                                attn_dropout = self.attn_dropout,
                                                relu_dropout = self.relu_dropout,
                                                res_dropout = self.res_dropout,
                                                attn_mask = self.attn_mask)
            self.layers.append(new_layer)

        self.register_buffer('version', torch.Tensor([2]))
        self.normalize = True
        if self.normalize:
            self.layer_norm = DynamicLayerNorm(embed_dim)

        super().__init__(embed_dim = self.embed_dim, layers = layers, 
                  SinusoidalPositionalEmbedding = self.embed_positions, layers_nn = self.layers, ln = self.layer_norm, 
                  attn_dropout = self.attn_dropout, relu_dropout = relu_dropout, res_dropout = res_dropout, 
                  embed_dropout=embed_dropout, attn_mask = self.attn_mask
        )
        
        self.active_layer_num = layers

    def forward(self, x_in, x_in_k = None, x_in_v = None, active_mask = [None]):
        # embed tokens and positions
        if not active_mask[0] is None:
            assert (x_in_k is None) and (x_in_v is None)
            self.embed_positions.embedding_dim = len(active_mask)
        else:
            self.embed_positions.embedding_dim = self.embed_dim
            
        x = self.embed_scale * x_in
        if self.embed_positions is not None:
            s = self.embed_positions(x.transpose(0, 1)[:, :, 0]).transpose(0, 1)   # Add positional embedding
            x += s
        x = F.dropout(x, p=self.dropout, training=self.training)

        if x_in_k is not None and x_in_v is not None:
            # embed tokens and positions    
            x_k = self.embed_scale * x_in_k
            x_v = self.embed_scale * x_in_v
            if self.embed_positions is not None:
                x_k += self.embed_positions(x_in_k.transpose(0, 1)[:, :, 0]).transpose(0, 1)   # Add positional embedding
                x_v += self.embed_positions(x_in_v.transpose(0, 1)[:, :, 0]).transpose(0, 1)   # Add positional embedding
            x_k = F.dropout(x_k, p=self.dropout, training=self.training)
            x_v = F.dropout(x_v, p=self.dropout, training=self.training)
        
        # encoder layers
        for i in range(self.active_layer_num):
            layer = self.layers[i]
            if x_in_k is not None and x_in_v is not None:
                x = layer(x, x_k, x_v)
            else:
                x = layer(x, active_mask = active_mask)   
        x = self.layer_norm(x, active_mask)
        return x

    """ To be Tested add """
    def get_active_subnet(self, active_layer_num, active_dimension, active_head_num, active_head_dim, active_mask = [None]):
        SinusoidalPositionalEmbedding = self.embed_positions
        if not active_mask[0] is None:
          SinusoidalPositionalEmbedding.embedding_dim = len(active_mask)
        layers_nn = [l.get_active_subnet(active_dimension, active_head_num, active_head_dim, active_mask) for l in self.layers[:active_layer_num]]
        ln = self.layer_norm.copy(active_mask = active_mask)
        sub_layer = TransformerEncoder(self.embed_dim, active_layer_num, 
                  SinusoidalPositionalEmbedding, layers_nn, ln, 
                  attn_dropout=self.attn_dropout, relu_dropout=self.relu_dropout, 
                  res_dropout=self.res_dropout, embed_dropout=self.embed_dropout, attn_mask=self.attn_mask)
        sub_layer = sub_layer.to(self.parameters().__next__().device)
        return sub_layer
    
    def set_active(self, active_layer_num, active_dimension, active_head_num, active_head_dim):
        self.active_layer_num = active_layer_num
        for i in range(self.active_layer_num):
            self.layers[i].set_active(active_dimension = active_dimension, active_head_dim = active_head_dim, active_head_num = active_head_num)

    def sort(self, sort_head = False, sort_head_dim = False, sort_dim_transformer_layer = False):
        for i in range(len(self.layers)):
            self.layers[i].sort(sort_head, sort_head_dim, sort_dim_transformer_layer)


class DynamicTransformerEncoderLayer(TransformerEncoderLayer):
    def __init__(self, embed_dim_in, head_dim, num_heads,  
                  attn_dropout=0.1, relu_dropout=0.1, res_dropout=0.1,
                  attn_mask=False):
        nn.Module.__init__(self)
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.embed_dim =  self.head_dim * self.num_heads
        self.embed_dim_in = embed_dim_in
        self.embed_dim_out = embed_dim_in

        self.attn_dropout = attn_dropout
        self.attn_mask = attn_mask
        self.relu_dropout = relu_dropout
        self.res_dropout = res_dropout
        self.normalize_before = True

        self.self_attn = DynamicMultiheadAttention(
            embed_dim_in = self.embed_dim_in,
            head_dim = self.head_dim,
            num_heads = self.num_heads,
            attn_dropout = self.attn_dropout
        )
        
        #dynamic part
        self.fc1 = DynamicLinear(self.embed_dim_in, 4 * self.embed_dim, bias = True)   # The "Add & Norm" part in the paper
        self.fc2 = DynamicLinear(4 * self.embed_dim, self.embed_dim_out, bias= True)

        self.layer_norms = nn.ModuleList([DynamicLayerNorm(self.embed_dim_in) for _ in range(2)])

        super(DynamicTransformerEncoderLayer, self).__init__(
            self.self_attn, self.fc1, self.fc2, self.layer_norms, 
            self.attn_dropout, self.relu_dropout, self.res_dropout, self.attn_mask
        )

        self._init_parameters()
        self.active_hidden_out_fc1 = 4 * self.embed_dim

    def _init_parameters(self):
        nn.init.xavier_uniform_(self.fc1.l.weight)
        nn.init.constant_(self.fc1.l.bias, 0.)
        nn.init.xavier_uniform_(self.fc2.l.weight)
        nn.init.constant_(self.fc2.l.bias, 0.)

    """To be Tested!"""
    def forward(self, x, x_k=None, x_v=None, active_mask = [None]):
        if active_mask[0] is not None:
          assert len(active_mask) == x.size()[-1]
        residual = x
        x = self.maybe_layer_norm(0, x, before=True, active_mask = active_mask)
        
        mask = buffered_future_mask(x, x_k) if self.attn_mask else None
        if x_k is None and x_v is None:
            x = self.self_attn(query = x, key = x, value = x, attn_mask = mask, active_mask = active_mask)
        else:
            x_k = self.maybe_layer_norm(0, x_k, before=True)
            x_v = self.maybe_layer_norm(0, x_v, before=True)
            x = self.self_attn(query = x, key = x_k, value = x_v, attn_mask = mask)
              
        x = F.dropout(x, p=self.res_dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(0, x, after=True, active_mask = active_mask)
        
        residual = x
        x = self.maybe_layer_norm(1, x, before=True, active_mask = active_mask)
        x = self.fc1(x, active_dim_in = None, active_dim_out = self.active_hidden_out_fc1, 
                    mask_out =[None], mask_in = active_mask)
        x = F.relu(x)
        x = F.dropout(x, p = self.relu_dropout, training = self.training)
        x = self.fc2(x, active_dim_out = None, active_dim_in = self.active_hidden_out_fc1, 
                    mask_out = active_mask, mask_in = [None])
        x = F.dropout(x, p=self.res_dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(1, x, after = True, active_mask = active_mask)
        return x

    """To be Tested!"""
    def sort_hidden_layers(self):
        # sort fc1 out dimension (0) and fc2 in dimension (1)
        importance = torch.sum(
            torch.abs(self.fc1.weight.data), 
            dim = 1,
            keepdim = False
        )
        sorted_importance, sorted_idx = torch.sort(importance, dim = 0, descending=True)

        #sort out dimension for fc1
        self.fc1.weight.data = torch.index_select(
            self.fc1.weight.data, 0, sorted_idx
        )
        self.fc1.bias.data = torch.index_select(
            self.fc1.bias.data, 0, sorted_idx
        )
        #sort in dimension for fc2
        self.fc2.weight.data = torch.index_select(
            self.fc2.weight.data, 1, sorted_idx
        )

        return sorted_idx

    """To be Tested!"""
    def get_active_subnet(self, active_dimension, active_head_num, active_head_dim, active_mask = [None]):

        self_attn = self.self_attn.get_active_subnet(active_head_dim, active_head_num, active_mask = active_mask)
        
        fc1 = self.fc1.copy(dim_in = None, dim_out = active_dimension, 
                          mask_in = active_mask, mask_out = [None])

        """ get subnet from fc2 """
        fc2 = self.fc2.copy(dim_out = None, dim_in = active_dimension, 
                          mask_out = active_mask, mask_in = [None])
                          
        """get subnet from ln"""
        layer_norms = [l.copy(active_mask) for l in self.layer_norms]
            
        sub_layer = TransformerEncoderLayer(
            self_attn, fc1, fc2, layer_norms,
            self.attn_dropout, self.relu_dropout, self.res_dropout, self.attn_mask
        )
        sub_layer = sub_layer.to(self.parameters().__next__().device)
        return sub_layer

    """To be Tested"""
    def maybe_layer_norm(self, i, x, before=False, after=False, active_mask = [None]):
        assert before ^ after
        if after ^ self.normalize_before:
            if not active_mask[0] is None:
              return self.layer_norms[i](x, active_mask = active_mask)
            else:
              return self.layer_norms[i](x)
        else:
            return x

    def set_active(self, active_dimension, active_head_num, active_head_dim):
       self.active_hidden_out_fc1 = active_dimension
       self.self_attn.set_active(active_head_dim = active_head_dim, active_num_heads = active_head_num)

    def sort(self, sort_head = False, sort_head_dim = False, sort_dim_transformer_layer = False):
        if sort_head:
            self.self_attn.sort_head()
        if sort_head_dim:
            self.self_attn.sort_hidden_layers()
        if sort_dim_transformer_layer:
            self.sort_hidden_layers()

