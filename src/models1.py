import torch
from torch import nn
import torch.nn.functional as F

from modules.transformer import TransformerEncoder
from modules.dynamic_transformer import DynamicTransformerEncoder

class MULTModel(nn.Module):
    def __init__(self, proj, trans, trans_mems, proj1, proj2, out_layer,
        origin_dimensions:list, dimension, 
        num_heads, head_dim, layers_hybrid_attn, layers_self_attn, attn_dropout:list, 
        relu_dropout, res_dropout, out_dropout, embed_dropout, attn_mask, output_dim):
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

        """ Shrinkable Hyperparameters
        in the parent graph we set up each modality with the same number of layers, hidden dimensions, head numbers and etc. 
        But the number of layers, head numbers, head dim for each modality are not required to be same during sampling!
        Output dimension of the temporal conv layers should always be the same """
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.layers_hybrid_attn = layers_hybrid_attn
        self.layers_self_attn = layers_self_attn
        self.modality_num = len(self.orig_dimensions)
        self.combined_dim = self.modality_num * (self.modality_num - 1) * self.d
        
        """ Temporal Convolutional Layers (None Shrinkable) """
        self.proj = proj
        self.proj = nn.ModuleList(self.proj)

        """ Crossmodal Attentions (Shrinkable) """
        self.trans = trans
        self.trans = nn.ModuleList(self.trans)

        """ Self Attentions (Shrinkable) """
        self.trans_mems = trans_mems
        self.trans_mems = nn.ModuleList(self.trans_mems)
         
        """ Projection Layers (Shrinkable) """
        self.proj1 = proj1
        self.proj2 = proj2
        self.out_layer = out_layer

   
    def forward(self, x):
        assert len(x) == self.modality_num # missing modality will be repalced by ones or zeros, can not be deleted
        x = [v.permute(0, 2, 1)for v in x]  # n_modalities * [batch_size, n_features, seq_len]

        proj_x = [self.proj[i](x[i]) for i in range(self.modality_num)]
        proj_x = torch.stack(proj_x)
        proj_x = proj_x.permute(0, 3, 1, 2)
        
        hs = []
        last_hs = []
        for i in range(self.modality_num):
            h = []
            for j in range(self.modality_num):
                if j != i:
                  h.append(self.trans[i][j](proj_x[i], proj_x[j], proj_x[j]))
            h = torch.cat(h, dim = 2)
            h = self.trans_mems[i](h)
            if type(h) == tuple:
                h = h[0]
            last_hs.append(h[-1])
        
        out = torch.cat(last_hs, dim=1)
        
        out_proj = self.proj2(
            F.dropout(F.relu(self.proj1(out)), p = self.out_dropout, training = self.training)
        )
        out_proj += out
        
        out = self.out_layer(out_proj)
        return out

