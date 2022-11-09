'''
Author: Jinguang Tong
Affliction: Australia National University, DATA61, Black Mountain
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.utils import Sine, sine_init

from .embedder import get_embedder

# This implementation is borrowed from IDR: https://github.com/lioryariv/idr
class SDFNetwork(nn.Module):
    def __init__(self, d_in, d_out, d_hidden, n_layers, skip_in=[4,], multires=0, bias=0.5, scale=1, geometric_init=False, weight_norm=False, inside_outside=False, omega=30, init_fn=None, act_fn=None):
        """
        Args:
            d_in ([type]): input dimension
            d_out ([type]): output dimension
            d_hidden ([type]): hidden layer dimension
            n_layers ([type]): number of hidden layers
            skip_in (list, optional): layers at which concatenates input. Defaults to [4,].
            multires (int, optional): number of frequency for position encoding. Defaults to 0.
            bias (float, optional): bias for linear layer initialization. Defaults to 0.5.
            scale (int, optional): [description]. Defaults to 1.
            geometric_init (bool, optional): [description]. Defaults to True.
            weight_norm (bool, optional): [description]. Defaults to True.
            inside_outside (bool, optional): Chage the sign of sdf value of inside and outside. Defaults to False.
        """
        super().__init__()

        # input dimension of each layer
        dims = [d_in] + [d_hidden for _ in range(n_layers)] + [d_out]

        self.embed_fn_fine = None
        
        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
            self.embed_fn_fine = embed_fn
            dims[0] = input_ch
        
        self.num_layers = len(dims)
        self.skip_in = skip_in
        self.scale = scale
        
        for l in range(0, self.num_layers-1):
            if l+1 in self.skip_in:
                out_dim = dims[l+1] - dims[0]
            else: 
                out_dim = dims[l+1]
                
            lin = nn.Linear(dims[l], out_dim)
            
            if init_fn is not None:
                eval(init_fn)(lin, is_first_layer=True if l == 0 else False, omega=omega)
            
            elif geometric_init:
                if l == self.num_layers - 2:
                    if not inside_outside:
                        torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=1e-4)
                        torch.nn.init.constant_(lin.bias, -bias)
                    else:
                        torch.nn.init.normal_(lin.weight, mean=-np.sqrt(np.pi) / np.sqrt(dims[l]), std=1e-4)
                        torch.nn.init.constant_(lin.bias, bias)
                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
            
            if weight_norm:
                lin = nn.utils.weight_norm(lin)
                
            setattr(self, 'lin' + str(l), lin)

        self.activation = eval(act_fn)(omega) if act_fn is not None else nn.Softplus(beta=100)
    
    def forward(self, inputs):
        inputs = inputs * self.scale
        if self.embed_fn_fine:
            inputs = self.embed_fn_fine(inputs)
        
        x = inputs
        for l in range(0, self.num_layers-1):
            lin = getattr(self, 'lin'+str(l))
            
            if l in self.skip_in:
                x = torch.cat([x, inputs], 1) / np.sqrt(2)
            
            x = lin(x)
            
            if l < self.num_layers - 2:
                x = self.activation(x)
        return torch.cat([x[:, :1] / self.scale, x[:, 1:]], dim=-1)
    
    def sdf(self, x):
        return self.forward(x)[:, :1]

    def sdf_hidden_appearance(self, x):
        return self.forward(x)
    
    def gradient(self, x):
        x.requires_grad_(True)
        y = self.sdf(x)
        d_output = torch.ones_like(y, requires_grad=False)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return gradients.unsqueeze(1)            
            
# This implementation is borrowed from IDR: https://github.com/lioryariv/idr
class RenderingNetwork(nn.Module):
    def __init__(self, d_feature, mode, d_in, d_out, d_hidden, n_layers, weight_norm=True, multires_view=0, squeeze_out=True):
        """
        Neural rendering network 
        
        Args:
            d_feature ([type]): [description]
            mdoe ([type]): [description]
            d_in ([type]): [description]
            d_out ([type]): [description]
            d_hidden ([type]): [description]
            n_layers ([type]): [description]
            weight_norm (bool, optional): [description]. Defaults to True.
            multires_view (int, optional): [description]. Defaults to 0.
            squeeze_out (bool, optional): [description]. Defaults to True.
        """
        super().__init__()
        
        self.mode = mode
        self.squeeze_out = squeeze_out
        dims = [d_in + d_feature] + [d_hidden for _ in range(n_layers)] + [d_out]
        
        self.embedview_fn = None
        if multires_view > 0:
            embedview_fn, input_ch = get_embedder(multires_view)
            self.embedview_fn = embedview_fn
            dims[0] += (input_ch - 3)
            
        self.num_layers = len(dims)

        for l in range(0, self.num_layers-1):
            out_dim = dims[l+1]
            lin = nn.Linear(dims[l], out_dim)
            
            if weight_norm:
                lin = nn.utils.weight_norm(lin)
            
            setattr(self, 'lin'+str(l), lin)
            
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, points, normals, view_dirs, feature_vectors):
        if self.embedview_fn:
            view_dirs = self.embedview_fn(view_dirs)
        
        rendering_input = None
        
        if self.mode == 'idr':
            rendering_input = torch.cat([points, view_dirs, normals, feature_vectors], dim=-1)
        elif self.mode == 'no_view_dir':
            rendering_input = torch.cat([points, normals, feature_vectors], dim=-1)
        elif self.mode == 'no_normal':
            rendering_input = torch.cat([points, view_dirs, feature_vectors], dim=-1)
        
        x = rendering_input
        
        for l in range(0, self.num_layers-1):
            lin = getattr(self, 'lin' + str(l))
            
            x = lin(x)
            
            if l < self.num_layers - 2:
                x = self.relu(x)
        if self.squeeze_out:
            x = torch.sigmoid(x)
        return x
        

class NeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_view=3, multires=0, multires_view=0, output_ch=4, skips=[4], use_viewdirs=False):
        super().__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_view = input_ch_view
        self.embed_fn = None
        self.embed_fn_view = None
        
        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=input_ch)  
            self.embed_fn = embed_fn
            self.input_ch = input_ch
            
        if multires_view > 0:
            embed_fn_view, input_ch_view = get_embedder(multires_view, input_dims=input_ch_view)
            self.embed_fn_view = embed_fn_view
            self.input_ch_view = input_ch_view
            
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        
        self.pts_linears = nn.ModuleList(
            [nn.Linear(self.input_ch, W)] + 
            [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + self.input_ch, W) for i in range(D-1)]
        )

        ### Implementation according to the official code release
        ### (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([nn.Linear(self.input_ch_view + W, W // 2)])
        
        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W // 2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)
    
    def forward(self, input_pts, input_views):
        if self.embed_fn:
            input_pts = self.embed_fn(input_pts)
        if self.embed_fn_view: 
            input_views = self.embed_fn_view(input_views)
        
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)
        
        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)
            
            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)
            
            rgb = self.rgb_linear(h)       
        else:
            output = self.output_linear(h)
            alpha, rgb = torch.split(output, [1, 3], dim=-1)

        return alpha, rgb
            

class SingleVarianceNetwork(nn.Module):
    def __init__(self, init_val):
        super().__init__()
        self.register_parameter('variance', nn.Parameter(torch.tensor(init_val)))
    
    def forward(self, x):
        return torch.ones([len(x), 1]) * torch.exp(self.variance * 10.0)
    
        

if __name__ == '__main__':
    sdf = SDFNetwork(3, 1, 200, 8)
    input = torch.rand(100, 3)
    grad = sdf.gradient(input)
    print(grad.norm(p=2, dim=-1))

    