import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.loss import L1Loss

class DynamicLinear(nn.Module):
    def __init__(self, dim_in, dim_out, bias):
        super().__init__()
        self.l = nn.Linear(dim_in, dim_out, bias)
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.bias = bias
        assert self.bias == True

    def forward(self, x, active_dim_in = None, active_dim_out = None, mask_in:list = [None], mask_out:list =[None]):
        weight = self.l.weight.contiguous()[:active_dim_out, :active_dim_in]
        bias = self.l.bias.contiguous()[:active_dim_out]
        if not (mask_in[0] is None):
            assert active_dim_in is None
            weight = torch.index_select(weight, 1, mask_in)
        if not (mask_out[0] is None):
            assert active_dim_out is None
            weight = torch.index_select(weight, 0, mask_out)
            bias = torch.index_select(bias, 0, mask_out)
        return F.linear(x, weight, bias)

    """ return part of the linear layer """
    def copy(self, dim_in = None, dim_out = None, mask_in = [None], mask_out = [None]):
        dim_in_ = self.dim_in if (dim_in == None and mask_in[0] is None) else max(dim_in if dim_in is not None else -1, len(mask_in))   
        dim_out_ = self.dim_out if (dim_out == None and mask_out[0] is None) else max(dim_out if dim_out is not None else -1, len(mask_out))
          
        L = nn.Linear(dim_in_, dim_out_).to(self.parameters().__next__().device)
        
        weight = self.l.weight.data
        bias = self.l.bias.data

        if not mask_in[0] is None:
            assert dim_in is None
            weight = torch.index_select(self.l.weight.data, 1, mask_in)
        
        if not mask_out[0] is None:
            assert dim_out is None
            weight = torch.index_select(self.l.weight.data, 0, mask_out)
            bias = torch.index_select(self.l.bias.data, 0, mask_out)
  
        L.weight.data.copy_(
          weight[:dim_out_, :dim_in_]
        )
        L.bias.data.copy_(
          bias[:dim_out_]
        )

        
        return L


class DynamicLayerNorm(nn.Module):
    def __init__(self, dim_in, dim_mask = None):
        super().__init__()
        self.ln = nn.LayerNorm(dim_in)
    def forward(self, x, active_mask = [None]):
        if not active_mask[0] is None:
            weight = torch.index_select(self.ln.weight.data, 0, active_mask)
            bias = torch.index_select(self.ln.bias.data, 0, active_mask)
            return F.layer_norm(x, (len(active_mask),), weight, bias)
        else:
            return self.ln(x)
    def copy(self, active_mask = [None]):
        if not active_mask[0] is None:
          target_ln = nn.LayerNorm(len(active_mask))
          target_ln.weight.data.copy_(torch.index_select(self.ln.weight.data, 0, active_mask))
          target_ln.bias.data.copy_(torch.index_select(self.ln.bias.data, 0, active_mask))
          return target_ln
        else:
          return self.ln
       
